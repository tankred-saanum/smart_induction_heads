import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
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

def get_chunks(A):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, (i*args.chunk_size):(i+1)*args.chunk_size, (j*args.chunk_size):(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--total_batch_size', default=1, type=int)
    parser.add_argument('--n_permute', default=3, type=int)
    parser.add_argument('--chunk_size', default=3, type=int)
    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--n_permute_primitive', default=3, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--seed', default=42, type=int)
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size//args.batch_size


    return args
args = get_config()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device='mps'
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto")
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads 

attn_heads = defaultdict(list)
all_chunk_ids =[]
accuracies = []

layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))

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
        output = model(batched_tokens, output_attentions=True)
    
    all_chunk_ids.append(chunk_ids)
    for layer in list(layer_dict.keys()):
        heads = layer_dict[layer]
        for head in heads:
            address = f'{layer}-{head}'
            attn_heads[address].append(output['attentions'][layer][:, head])

args.module='heads'
decoding_accs = torch.load(f'data/one_back_scores/markov{args.markov_order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
learning_scores = torch.load(f'data/learning_scores/markov{args.markov_order}/{args.model_name.split("/")[-1]}/learning_scores.pt')
induction_scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')

ldict = create_LH_dict(decoding_accs, threshold=0.90)
best_address, worst_address = get_best_and_worst(learning_scores, induction_scores=induction_scores, threshold=0.4)
best_address
worst_address


ldict = {best_address[0]:[best_address[1]], worst_address[0]:[worst_address[1]]}

if args.markov_order==2:
    grid_interval = args.chunk_size
    f, ax = plt.subplots(2, 2, figsize=(10, 8))
    lwd=0.75
    perm_id
    id_to_letter = {i: chr(945 + i) for i in range(args.n_permute)}
    id_to_alphabet = {t.item(): chr(97 + i) for i, t in enumerate(batched_tokens.unique())}
    greek_labels = [f"${id_to_letter[p.item()]}$" for p in perm_id]
    
    alpabet_labels = [id_to_alphabet[token.item()] for token in batched_tokens[0]]
    fs=15
    for i, l in enumerate(ldict.keys()):
        for h in ldict[l]:
            attn = attn_heads[f'{l}-{h}'][0][0].cpu().float()
            pooled = get_chunks(attn.unsqueeze(0)).squeeze(0)

            if i ==0:
                ax[0, i].set_title(f'Head {l} - {h}\n\nAdaptive induction head')
            else:
                ax[0, i].set_title(f'Head {l} - {h}\n\nStatic induction head')
            ax[0, i].imshow(pooled, cmap='copper')
            ax[0, i].set_xticks(torch.arange(len(greek_labels)))
            ax[0, i].set_yticks(torch.arange(len(greek_labels)))
            ax[0, i].set_xticklabels(greek_labels, size=fs)
            ax[0, i].set_yticklabels(greek_labels, size=fs)
            
            ax[1, i].imshow(attn, cmap='copper')
            ax[1, i].set_xticks(torch.arange(0, len(alpabet_labels), args.chunk_size)+1.0)
            ax[1, i].set_yticks(torch.arange(0, len(alpabet_labels), args.chunk_size)+1.0)
            ax[1, i].set_xticklabels(greek_labels, size=fs)
            ax[1, i].set_yticklabels(greek_labels, size=fs)
            
            for j in range(grid_interval, attn.shape[1], grid_interval):
                ax[1, i].axvline(x=j-0.5, color='white', linewidth=lwd)

            for j in range(grid_interval, attn.shape[0], grid_interval):
                ax[1, i].axhline(y=j-0.5, color='white', linewidth=lwd)

    plt.savefig('figures/induction_heads_visualization_order=2.png', bbox_inches='tight')
    plt.show()



if args.markov_order==3:
    grid_interval = args.chunk_size*args.n_permute_primitive
    f, ax = plt.subplots(2, 2, figsize=(10, 8))
    lwd=0.75
    id_to_letter = {i: chr(945 + i) for i in range(args.n_permute)}
    greek_letters = ['\u03A9', '\u03C8', '\u03C6'] # Ω, ψ, φ
    id_to_letter = {i: greek_letters[i] for i in range(args.n_permute)}
    id_to_alphabet = {t.item(): chr(945 + i) for i, t in enumerate(batched_tokens.unique())}
    greek_labels = [f"${id_to_letter[p.item()]}$" for p in perm_id]
    #[f"${id_to_letter[id]}$" for id in chunk_ids]
    alpabet_labels = [id_to_alphabet[token.item()] for token in batched_tokens[0]]
    fs=15
    for i, l in enumerate(ldict.keys()):
        for h in ldict[l]:
            attn = attn_heads[f'{l}-{h}'][0][0].cpu().float()
            #pooled = get_chunks(attn.unsqueeze(0)).squeeze(0)
            pooled = get_chunks_3rd_order_uniform(attn.unsqueeze(0)).squeeze(0)
            #attn = get_chunks(attn.unsqueeze(0)).squeeze(0)
            if i ==0:
                ax[0, i].set_title(f'Head {l} - {h}\n\nOff-diagonal head')
            else:
                ax[0, i].set_title(f'Head {l} - {h}\n\nN-tokens-back\nhead')
            ax[0, i].imshow(pooled, cmap='copper')
            ax[0, i].set_xticks(torch.arange(len(greek_labels)))
            ax[0, i].set_yticks(torch.arange(len(greek_labels)))
            ax[0, i].set_xticklabels(greek_labels, size=fs)
            ax[0, i].set_yticklabels(greek_labels, size=fs)
            
            ax[1, i].imshow(attn, cmap='copper')
            ax[1, i].set_xticks(torch.arange(0, attn.size(0), args.chunk_size*args.n_permute_primitive)+args.chunk_size+1)
            ax[1, i].set_yticks(torch.arange(0, attn.size(0), args.chunk_size*args.n_permute_primitive)+args.chunk_size+1)
            ax[1, i].set_xticklabels(greek_labels, size=fs)
            ax[1, i].set_yticklabels(greek_labels, size=fs)
            
            for j in range(grid_interval, attn.shape[1], grid_interval):
                ax[1, i].axvline(x=j-0.5, color='white', linewidth=lwd)

            for j in range(grid_interval, attn.shape[0], grid_interval):
                ax[1, i].axhline(y=j-0.5, color='white', linewidth=lwd)

    plt.savefig('figures/one_back_heads_visualization_order=3.png', bbox_inches='tight')
    plt.show()
