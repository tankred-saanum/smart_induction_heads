import torch
import sys
sys.path.insert(0, "..")


from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from nnsight import LanguageModel
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from utils import first_order_markov_sequence, second_order_markov_sequence, third_order_markov_sequence, unique_second_order_markov_sequence, unique_third_order_markov_sequence, create_LH_dict
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

    parser.add_argument('--n_reps', default=24, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--total_batch_size', default=1, type=int)
    parser.add_argument('--n_permute', default=3, type=int)
    parser.add_argument('--ablate', default=0, type=int)
    parser.add_argument('--chunk_size', default=4, type=int)
    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size//args.batch_size
    if args.markov_order==3:
        args.chunk_size = args.chunk_size//2

    return args
args = get_config()

device='mps'
model = LanguageModel(args.model_name, device_map="auto")#AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = model.tokenizer#AutoTokenizer.from_pretrained(args.model_name, device_map="auto")
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config

attn_heads = defaultdict(list)
all_chunk_ids =[]
accuracies = []

layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))



def unique_second_order_markov_sequence(tokens, args, return_perms=False):
    """
    Generates a sequence of tokens based on a second-order Markov structure.

    Returns:
        all_tokens (Tensor): The concatenated sequence of all tokens.
        chunk_id (Tensor): A matrix indicating which tokens belong to the same original chunk.
        permuted_sequence (Tensor): The sequence of indices of the permutations used.
        chunked_sequence (Tensor): The sequence of permutations (chunks) as they appear.
    """
    perms = []
    used_perms_indices = set()
    # Generate unique permutations of the input tokens
    while len(perms) < args.n_permute:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # Create a random sequence of these unique permutations
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    chunked_sequence_list = []
    for seq_id in permuted_sequence:
        chunked_sequence_list.append(perms[seq_id])

    # Stack the list of chunks into a single tensor
    chunked_sequence = torch.stack(chunked_sequence_list, dim=0)
    
    # Flatten the sequence for other uses
    all_tokens = torch.cat(chunked_sequence_list, dim=0)
    
    # Calculate chunk_id for identifying tokens from the same original permutation instance
    chunk_id = (torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        return all_tokens, chunk_id, permuted_sequence
    return all_tokens, chunk_id


save_layers = [2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 24, 26]

score_arr =  torch.load(f'data/one_back_scores/markov2/{args.model_name.split("/")[-1]}/heads/decoding_accuracies.pt')
score_dict = create_LH_dict(score_arr, threshold=0.9)
ablate_dict = score_dict

representation_dict = {}
for iter in range(args.iters):
    print(iter/args.iters)
    batched_tokens = []
    chunk_ids = []

    for _ in range(args.batch_size):
        tokens = torch.randint(vocab_size, (args.chunk_size, ))
        if args.markov_order == 2:
            all_tokens, chunk_id, perm_sequence = unique_second_order_markov_sequence(tokens=tokens, args=args, return_perms=True)
            
        elif args.markov_order == 3:
            all_tokens, chunk_id = unique_third_order_markov_sequence(tokens, args)
            
        batched_tokens.append(all_tokens)
        chunk_ids.append(chunk_id)

    batched_tokens = torch.stack(batched_tokens, dim=0).to(device)
    chunk_ids = torch.stack(chunk_ids, dim=0)

    with torch.no_grad():
        with model.trace(batched_tokens):
            if args.ablate:
                for layer in range(config.num_hidden_layers):
                    if layer in list(ablate_dict.keys()):
                        heads = ablate_dict[layer]
            
                        o_proj_in = model.model.layers[layer].self_attn.o_proj.input
                        o_proj_in = o_proj_in.view(o_proj_in.size(0), o_proj_in.size(1), n_heads, o_proj_in.size(-1)//n_heads)
                        o_proj_in[:, :, heads] *= 0
                        o_proj_in = o_proj_in.view(o_proj_in.size(0), o_proj_in.size(1), -1)
                        model.model.layers[layer].self_attn.o_proj.input = o_proj_in
                        
            for layer in save_layers:
                repr = model.model.layers[layer].output.save()
                representation_dict[layer] = repr
                
    

from sklearn.manifold import MDS
colors= perm_sequence.repeat_interleave(args.chunk_size)
for layer in save_layers:
    x= representation_dict[layer][0].detach().cpu().squeeze(0)
    x = x[len(x)//3:]
    z = MDS(n_components=2).fit_transform(x)
    
    #plt.scatter(z[-250:, 0], z[-250:, 1], c=colors[-250:])
    plt.scatter(z[:, 0], z[:, 1], c=colors[-len(x):])
    plt.title(f'Layer: {layer}')
    plt.show()