import torch
import sys
sys.path.insert(0, '/Users/tankredsaanum/Documents/smart_induction_heads')

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
from src.utils import first_order_markov_sequence, second_order_markov_sequence, third_order_markov_sequence, unique_second_order_markov_sequence, unique_third_order_markov_sequence, create_LH_dict, create_random_dict

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

    parser.add_argument('--n_reps', default=8, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--total_batch_size', default=32, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
    parser.add_argument('--ablation_style', default='one_back', type=str)   
    args, _ = parser.parse_known_args()
    if args.ablation_style=='random' or args.ablation_style =='random_induction':
        # set a minimal batch size when random ablation so we get a new set of heads for each batch
        args.batch_size=1

    args.iters = args.total_batch_size//args.batch_size

    if args.markov_order==3:
        args.chunk_size = args.chunk_size//2

    return args
args = get_config()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device='mps'
model = LanguageModel(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = model.tokenizer
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config
layerwise_logliks = defaultdict(list)
layerwise_accuracies = defaultdict(list)
attn_heads = defaultdict(list)
all_chunk_ids =[]



order='markov2' if args.markov_order==2 else 'markov3'
layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))

if args.ablation_style == 'induction':
    args.threshold=0.4 # hard code threshold
    score_arr = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')

elif args.ablation_style == 'learning':
    args.threshold=0.4 # fix these for simplicity
    score_arr = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/learning_scores.pt')

elif args.ablation_style == 'one_back':
    # always load markov 2 decodability to make analyses comparable
    score_arr =  torch.load(f'data/one_back_scores/markov2/{args.model_name.split("/")[-1]}/heads/decoding_accuracies.pt')
elif args.ablation_style == 'random':
    score_arr =  torch.load(f'data/one_back_scores/markov2/{args.model_name.split("/")[-1]}/heads/decoding_accuracies.pt')
elif args.ablation_style == 'random_induction':
    args.threshold=0.4 # hard code threshold
    score_arr = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')


if args.ablation_style != 'random':
    score_dict = create_LH_dict(score_arr, threshold=args.threshold)
    ablate_dict = score_dict
    print(ablate_dict)


accuracies = []

for iter in range(args.iters):
    print(iter/args.iters)
    batched_tokens = []
    chunk_ids = []
    if args.ablation_style == 'random':
        score_dict = create_random_dict(score_arr, threshold=args.threshold, pool_threshold=0.55)
        ablate_dict = score_dict
        print(ablate_dict)
    elif args.ablation_style == 'random_induction':
        score_dict = create_random_dict(score_arr, threshold=args.threshold, pool_threshold=args.threshold)
        ablate_dict = score_dict
        print(ablate_dict)
    for _ in range(args.batch_size):
        tokens = torch.randint(vocab_size, (args.chunk_size, ))
        if args.markov_order == 2:
            all_tokens, chunk_id = unique_second_order_markov_sequence(tokens, args)
            
        elif args.markov_order == 3:
            all_tokens, chunk_id = unique_third_order_markov_sequence(tokens, args)
            
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


save_dir = Path(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}')
save_dir.mkdir(parents=True, exist_ok=True)
learning_scores = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
#induction_layers = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
induction_scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')
orig_learning_scores = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/learning_scores.pt')

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
        is_induction=induction_scores[layer, head] > 0.4
        
        if orig_learning_scores[layer, head] > 0.4 or is_induction:
            torch.save(head_accs,f'{save_dir}/{address}_accs.pt')
            
torch.save(learning_scores, f'{save_dir}/learning_scores.pt')
torch.save(accuracies, f'{save_dir}/model_accs.pt')
torch.save(args, f'{save_dir}/args.pt')   
