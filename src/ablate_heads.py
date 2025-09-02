import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
import nnsight
from nnsight import LanguageModel
from pathlib import Path
from collections import OrderedDict
from utils import second_order_markov_sequence, third_order_markov_sequence

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



def get_config():
    parser = ArgumentParser()

    parser.add_argument('--n_pad_repetitions', default=4, type=int)
    parser.add_argument('--n_reps', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=64, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cmap', default='cividis', type=str)
    parser.add_argument('--ablation_style', default='induction', type=str, choices=['induction', 'learning', 'one_back_identity'])   
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
    args, _ = parser.parse_known_args()

    return args
args = get_config()
torch.manual_seed(args.seed)

args.iters = args.total_batch_size//args.batch_size
device='mps'
model = LanguageModel(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = model.tokenizer
model.model.norm
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config
layerwise_logliks = defaultdict(list)
layerwise_accuracies = defaultdict(list)
attn_heads = defaultdict(list)
all_chunk_ids =[]




layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))

if args.ablation_style == 'induction':
    induction_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')

    ablate_dict = induction_dict
if args.ablation_style == 'learning':
    pass

# sorted_layers = sorted(layer_dict.keys())
# ablate_list = []
# ablate_dict = defaultdict(list)
# for layer in sorted_layers:
#     for val in layer_dict[layer]:
#         ablate_list.append((layer, val))
        
# ablate_list
# num_ablate=6
# for i in range(num_ablate):
#     layer, head = ablate_list[i]
#     ablate_dict[layer].append(head)

# print(ablate_dict)
# ablate_dict = layer_dict
# ablate_dict[5] = [9, 14]
# ablate_dict[6] = [6]
# ablate_dict[17] = [14, 10]
# #ablate_dict = {5:[0, 1, 2, 3, 4, 5], 17: [0, 1, 2, 3, 4], 6:[0, 1, 2, 3, 4], 7:[0, 1, 2]}
# #ablate_dict = {5:[9, 14], 17:[14, 10], 6:[6], 7:[3]}
#ablate_dict = {4:[4], 6:[5], 8:[4], 10:[0], 12:[6], 18:[6]}
#ablate_dict = layer_dict
#print(layer_dict)
accuracies = []
for iter in range(args.iters):
    print(iter/args.iters)
    batched_tokens = []
    chunk_ids = []
    for _ in range(args.batch_size):
        tokens = torch.randint(vocab_size, (args.chunk_size, ))
        if args.markov_order == 2:
            all_tokens, chunk_id = second_order_markov_sequence(tokens, args)
            
        elif args.markov_order == 3:
            all_tokens, chunk_id = third_order_markov_sequence(tokens, args)
            
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
    for layer in list(layer_dict.keys()):
        heads = layer_dict[layer]
        for head in heads:
            address = f'{layer}-{head}'
            attn_heads[address].append(output['attentions'][layer][:, head])
        
    #for head in heads:

        
    # compute model accuracy
    logits = output['logits']
    
    pred = logits.argmax(dim=-1)
    acc = (pred[:, :-1] == batched_tokens[:, 1:]).float()
    accuracies.append(acc)
             
all_chunk_ids= torch.cat(all_chunk_ids, dim=0)
accuracies = torch.cat(accuracies, dim=0).cpu().float()

order='markov2' if args.markov_order==2 else 'markov3'
save_dir = Path(f'data/ablated_learning_scores/{order}/{args.model_name.split("/")[-1]}')
save_dir.mkdir(parents=True, exist_ok=True)
learning_scores = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
induction_layers = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
# before we aggregate attention by chunk size, we need to change the chunk size argument if we are in a 3rd order markov process
# this is because the real chunk size is actually args.chunk_size *args.n_permute_primitive
args.chunk_size = args.chunk_size * args.n_permute_primitive if args.markov_order == 3 else args.chunk_size
for layer in list(layer_dict.keys()):
    heads = layer_dict[layer]
    for head in heads:
        address = f'{layer}-{head}'
        attn_matrices = torch.cat(attn_heads[address], dim=0)
        if order == 'markov2':
            pooled = get_chunks(attn_matrices)
        else:
            pooled = get_chunks_3rd_order(attn_matrices)

        
        head_accs = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps)
        for i in range(1, pooled.size(1)):
            row_ideal = all_chunk_ids[:, i, :i]
            row_model = pooled[:, i, :i]
            most_attn_idx = row_model.argmax(dim=1)
            score = row_ideal[torch.arange(args.total_batch_size), most_attn_idx]
            #print(score.shape, 'score!')
            
            head_accs[:, i] = score
            
        learning_score = head_accs.mean(dim=0)[-10:].mean()
        learning_scores[layer, head] = learning_score
        is_induction=False
        if layer in list(induction_layers.keys()):
            if head in induction_layers[layer]:
                is_induction=True
        if learning_score > args.threshold or is_induction:
            torch.save(head_accs,f'{save_dir}/{address}_accs.pt')
            
torch.save(learning_scores, f'{save_dir}/learning_scores.pt')
torch.save(accuracies, f'{save_dir}/model_accs.pt')
torch.save(args, f'{save_dir}/args.pt')   
# order='markov2' if args.markov_order==2 else 'markov3'
# all_chunk_ids= torch.cat(all_chunk_ids, dim=0)
# accuracies = torch.cat(accuracies, dim=0).cpu().float()
# save_dir = Path(f'data/learning_scores_ablated/{order}/{args.model_name.split("/")[-1]}')
# save_dir.mkdir(parents=True, exist_ok=True)
# args.chunk_size = args.chunk_size * args.n_permute_primitive if args.markov_order == 3 else args.chunk_size
# for layer in list(induction_dict.keys()):
#     heads = induction_dict[layer]
#     for head in heads:
#         address = f'{layer}-{head}'
#         attn_matrices = torch.cat(attn_heads[address], dim=0)
#         if order == 'markov2':
#             pooled = get_chunks(attn_matrices)
#         else:
#             pooled = get_chunks_3rd_order(attn_matrices)

        
#         head_accs = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps)
#         for i in range(1, pooled.size(1)):
#             row_ideal = all_chunk_ids[:, i, :i]
#             row_model = pooled[:, i, :i]
#             most_attn_idx = row_model.argmax(dim=1)
#             score = row_ideal[torch.arange(args.total_batch_size), most_attn_idx]
#             #print(score.shape, 'score!')
            
#             head_accs[:, i] = score
            
        
#         torch.save(head_accs,f'{save_dir}/{address}_accs.pt')
# torch.save(accuracies, f'{save_dir}/model_accs.pt')


# for layer in list(layer_dict.keys()):
#     heads = layer_dict[layer]
#     for head in heads:
#         address = f'{layer}-{head}'
#         attn_matrices = torch.cat(attn_heads[address], dim=0)
#         attn_matrices.shape
#         pooled = get_chunks(attn_matrices)
#         pooled.shape
        
#         errors = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps)
#         for i in range(1, pooled.size(1)):
#             row_ideal = all_chunk_ids[:, i, :i]
#             row_model = pooled[:, i, :i]
#             most_attn_idx = row_model.argmax(dim=1)
#             score = row_ideal[torch.arange(args.total_batch_size), most_attn_idx]
#             #print(score.shape, 'score!')
            
#             errors[:, i] = score
            
#         ax.plot(errors.mean(dim=0), linewidth=3, label=f'LH: {address}')
# plt.tick_params(labelsize=12)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.ylabel('Attn Accuracy', size=14)
# plt.xlabel('Repetition', size=14)
# plt.ylim([0, 1.1])
# plt.legend()
# plt.savefig('figures/icl_heads_ablated2.png', dpi=200)
# plt.show()
# if True:
#     cmap = plt.cm.viridis
#     colors = [cmap(i / (config.num_hidden_layers - 1)) for i in range(config.num_hidden_layers)]

#     layer=config.num_hidden_layers-1
#     layer_accs = torch.cat(layerwise_accuracies[layer], dim=0)
#     #print(layer_accs.shape)
#     layer_accs = layer_accs.mean()
#     print(layer_accs)
    #layer_accs = layer_accs.view(-1, args.chunk_size)
    # plt.plot(layer_accs.mean(dim=0).float().cpu(), color=colors[layer])
    # plt.tick_params(labelsize=12)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.ylabel('Model accuracy', size=14)
    # plt.xlabel('Repetition', size=14)
    # plt.ylim([0, 1.1])
    # plt.show()

