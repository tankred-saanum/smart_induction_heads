import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from nnsight import LanguageModel
from transformers import PretrainedConfig

sys.path.insert(0, "..")


from argparse import ArgumentParser
from collections import defaultdict

import matplotlib as mpl

from utils import (
    create_LH_dict,
    get_best_and_worst,
    unique_second_order_markov_sequence,
    unique_third_order_markov_sequence,
)

mpl.rcParams['mathtext.fontset'] = 'cm'
from matplotlib.lines import Line2D


def get_chunks(A):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, (i*args.chunk_size):(i+1)*args.chunk_size, (j*args.chunk_size):(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    return B

def get_induction_head_acc(A, seq):
    B = torch.zeros(args.total_batch_size, A.size(1))
    for i in range(A.size(1)):
        tokens = seq[:, i]
        if i < args.chunk_size:
            continue
        most_attn_idx = torch.argmax(A[:, i], dim=-1)
        predecessor = seq[torch.arange(args.total_batch_size), most_attn_idx-1]
        is_successor = (predecessor == tokens).float()
        B[:, i] = is_successor
    return B


def get_chunks_3rd_order(A):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            rows = A[:, (i*args.chunk_size):(i+1)*args.chunk_size, :]
            transition_idx = torch.arange(1, args.chunk_size+1)
            mask = transition_idx % (args.chunk_size//args.n_permute_primitive) == 0
            mask[-1] = False
            rows = rows[:, mask]
            patch_score = rows[:, :, (j*args.chunk_size):(j+1)*args.chunk_size]
            
            B[:, i, j] = patch_score.reshape(args.total_batch_size, -1).mean(dim=-1)
            
    return B

def get_chunks_3rd_order_uniform(A):
    higher_order_chunk_size = args.chunk_size*args.n_permute_primitive
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, (i*higher_order_chunk_size):(i+1)*higher_order_chunk_size, (j*higher_order_chunk_size):(j+1)*higher_order_chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    

    return B

def unique_second_order_markov_sequence(tokens, args, return_perms=False):

    perms = []
    used_perms_indices = set()
    
    while len(perms) < args.n_permute:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    chunked_sequence_list = []
    for seq_id in permuted_sequence:
        chunked_sequence_list.append(perms[seq_id])

    
    chunked_sequence = torch.stack(chunked_sequence_list, dim=0)
    
    
    all_tokens = torch.cat(chunked_sequence_list, dim=0)
    
    chunk_id = (torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        return all_tokens, chunk_id, permuted_sequence
    return all_tokens, chunk_id

def unique_third_order_markov_sequence(tokens, args, return_perms=False):

    
    perms = []
    used_perms_indices = set()
    
    while len(perms) < args.n_permute_primitive:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    
    perms2 = []
    primitive_compositions = [] 
    used_perms2_indices = set()
    while len(perms2) < args.n_permute:
        perm_idx = torch.randperm(args.n_permute_primitive)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms2_indices:
            used_perms2_indices.add(perm_idx_tuple)
            primitive_compositions.append(perm_idx)
            _perm = torch.cat([perms[idx] for idx in perm_idx], dim=0)
            perms2.append(_perm)

   
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    high_order_permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    high_order_chunked_list = []
    primitive_permuted_list = []
    for seq_id in high_order_permuted_sequence:
        high_order_chunked_list.append(perms2[seq_id])
        primitive_permuted_list.append(primitive_compositions[seq_id])

    
    all_tokens = torch.cat(high_order_chunked_list, dim=0)
    
    
    chunk_id = (torch.cdist(high_order_permuted_sequence.unsqueeze(-1).float(), high_order_permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        return all_tokens, chunk_id, high_order_permuted_sequence
    return all_tokens, chunk_id
    
def get_config():
    parser = ArgumentParser()

    parser.add_argument('--n_reps', default=4, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--total_batch_size', default=4, type=int)
    parser.add_argument('--n_permute', default=3, type=int)
    parser.add_argument('--chunk_size', default=3, type=int)
    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--n_permute_primitive', default=3, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--seed', default=42, type=int)
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size//args.batch_size
    # if args.markov_order==3:
    #     args.chunk_size = args.chunk_size//2

    return args
args = get_config()

device='mps'
model = LanguageModel(args.model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
tokenizer = model.tokenizer
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config


layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))
    
args.ablation_style='one_back'
if args.ablation_style == 'one_back':
    # always load markov 2 decodability to make analyses comparable
    score_arr =  torch.load(f'data/one_back_scores/markov2/{args.model_name.split("/")[-1]}/heads/decoding_accuracies.pt')
if args.ablation_style != 'random':
    score_dict = create_LH_dict(score_arr, threshold=0.95)
    ablate_dict = score_dict
    print(ablate_dict)





# create figure here
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
lwd=0.75
colors = ['#8a2f08', '#2d7acc']
fs=12
for condition in ['normal', 'ablate']:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if condition == 'ablate':
        ablate_dict = {13:[4]}
        style = '--'
    else:
        ablate_dict = {}
        style = '-'
        label_suffix = 'Context match ablation'
        
    all_toks = []
    all_chunk_ids =[]
    accuracies = []
    

    attn_heads = defaultdict(list)
    for iter in range(args.iters):
        print(iter/args.iters)
        batched_tokens = []
        chunk_ids = []

        for _ in range(args.batch_size):
            tokens = torch.randint(vocab_size, (args.chunk_size, ))
            if args.markov_order == 2:
                all_tokens, chunk_id, perm_id = unique_second_order_markov_sequence(tokens, args, return_perms=True)
                
            elif args.markov_order == 3:
                all_tokens, chunk_id, perm_id = unique_third_order_markov_sequence(tokens, args, return_perms=True)
                
            batched_tokens.append(all_tokens)
            chunk_ids.append(chunk_id)


        batched_tokens = torch.stack(batched_tokens, dim=0).to(device)
        chunk_ids = torch.stack(chunk_ids, dim=0)

        with torch.no_grad():
            with model.trace(batched_tokens, output_attentions=True):

                for layer in range(config.num_hidden_layers):
                    if layer in list(ablate_dict.keys()):
                        heads = ablate_dict[layer]
            
                        o_proj_in = model.model.layers[layer].self_attn.o_proj.input
                        o_proj_in = o_proj_in.view(o_proj_in.size(0), o_proj_in.size(1), n_heads, o_proj_in.size(-1)//n_heads)
                        o_proj_in[:, :, heads] *= 0
                        o_proj_in = o_proj_in.view(o_proj_in.size(0), o_proj_in.size(1), -1)
                        model.model.layers[layer].self_attn.o_proj.input = o_proj_in

                output = model.output.save()
        
        all_chunk_ids.append(chunk_ids)
        all_toks.append(batched_tokens)
        for layer in list(layer_dict.keys()):
            heads = layer_dict[layer]
            for head in heads:
                address = f'{layer}-{head}'
                attn_heads[address].append(output['attentions'][layer][:, head])

    args.module='heads'

        
    decoding_accs = torch.load(f'data/one_back_scores/markov{args.markov_order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
    learning_scores = torch.load(f'data/learning_scores/markov{args.markov_order}/{args.model_name.split("/")[-1]}/learning_scores.pt')
    induction_scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')
    all_chunk_ids= torch.cat(all_chunk_ids, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    score_dict = create_LH_dict(learning_scores, threshold=0.85)
    best_address, worst_address = get_best_and_worst(learning_scores, induction_scores=induction_scores, threshold=0.4)
    best_address = [14, 3]
    ldict = {best_address[0]:[best_address[1]]}
    batched_tokens.unique().tolist()
    batched_tokens[0].tolist()

    grid_interval = args.chunk_size

    for i, l in enumerate(ldict.keys()):
        for h in ldict[l]:
            attn_matrices = torch.cat(attn_heads[f'{l}-{h}'], dim=0)
            #attn = attn_heads[][0][0].cpu().float()
            pooled = get_chunks(attn_matrices)
            head_accs = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps)
            for i in range(1, pooled.size(1)):
                row_ideal = all_chunk_ids[:, i, :i]
                row_model = pooled[:, i, :i]
                most_attn_idx = row_model.argmax(dim=1)
                score = row_ideal[torch.arange(args.total_batch_size), most_attn_idx]
                #print(score.shape, 'score!')
                
                head_accs[:, i] = score
            #print(head_accs)
            induction_accs = get_induction_head_acc(attn_matrices, seq=all_toks)

            attn = attn_matrices[0].float().cpu()
            headmap_idx = 0 if condition!= 'ablate' else 1
            ax[headmap_idx].imshow(attn, cmap='copper')
 
            start_chr = 97
            token_to_letter = {token: chr(start_chr+i) for i, token in enumerate(all_toks[0].unique().tolist())}
            seq_letters = [token_to_letter[token] for token in all_toks[0].tolist()]
            ax[headmap_idx].set_xticks(torch.arange(len(seq_letters)))
            ax[headmap_idx].set_yticks(torch.arange(len(seq_letters)))
            ax[headmap_idx].set_xticklabels([f"${letter}$" for letter in seq_letters], fontsize=fs-5)
            ax[headmap_idx].set_yticklabels([f"${letter}$" for letter in seq_letters], fontsize=fs-5, rotation=0)
            reps = torch.arange(args.n_reps*args.n_permute)
            #label1='Matching context' if condition =='normal'
            ax[2].plot(reps, head_accs.mean(dim=0)*100, color=colors[0], linestyle=style)
            ax[2].plot(torch.arange(args.chunk_size*args.n_reps*args.n_permute)/args.chunk_size, induction_accs.mean(dim=0)*100, color=colors[1], linestyle=style)
            #ax[2].plot(torch.arange(args.chunk_size*args.n_reps*args.n_permute)/args.chunk_size, torch.ones_like(induction_accs.mean(dim=0))*(100/3), color='gray', linestyle='--')
            
            ax[2].set_title('Attention accuracy (%)')
            ax[2].set_ylim([0, 101])
            ax[2].set_xlabel('Repetition')
            if headmap_idx == 1:
                
                ax[headmap_idx].set_title('Context ablation')
            else:
                ax[headmap_idx].set_title('No ablation')
                
            for j in range(grid_interval, attn.shape[1], grid_interval):
                ax[headmap_idx].axvline(x=j-0.5, color='white', linewidth=lwd)

            for j in range(grid_interval, attn.shape[0], grid_interval):
                ax[headmap_idx].axhline(y=j-0.5, color='white', linewidth=lwd)
                
                
            # Create custom legend elements
            # Color legend elements
            color_elements = [

            ]

            # Linestyle legend elements
            legend_elements = [
                Line2D([0], [0], color=colors[1], linewidth=2, label='Successor tokens'),
                Line2D([0], [0], color=colors[0], linewidth=2, label='Correct context'),
                Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='No ablation'),
                Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Context ablation')
            ]

            
            ax[2].legend(handles=legend_elements, 
            loc='lower right', 
            ncol=2,  # Display in 4 columns (horizontal layout)
            #bbox_to_anchor=(0.75, -0.18),  # Center horizontally, position at bottom
            frameon=True,
            fontsize=8,           # Control text size
            markerscale=1.75,       # Control line/marker size  
            columnspacing=1,     # Space between columns
            handlelength=1.75,      # Length of legend lines
            handletextpad=0.3)

plt.savefig('figures/induction_heads_visualization_order=2_ablated.png', bbox_inches='tight')
plt.show()