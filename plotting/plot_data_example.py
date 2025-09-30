import sys

sys.path.insert(0, "..")
from argparse import ArgumentParser
from collections import defaultdict

import torch
from matplotlib import pyplot as plt
from transformers import PretrainedConfig
from utils import (
    second_order_markov_sequence,
    third_order_markov_sequence,
)


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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--total_batch_size', default=1, type=int)
    parser.add_argument('--n_permute', default=3, type=int)
    parser.add_argument('--chunk_size', default=3, type=int)
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
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config

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
            all_tokens, chunk_id = second_order_markov_sequence(tokens, args)
            
        elif args.markov_order == 3:
            all_tokens, chunk_id = third_order_markov_sequence(tokens, args)
            
        batched_tokens.append(all_tokens)
        chunk_ids.append(chunk_id)

    batched_tokens = torch.stack(batched_tokens, dim=0).to(device)
    chunk_ids = torch.stack(chunk_ids, dim=0)
    

batched_tokens = batched_tokens.squeeze(0).cpu().detach()
unique = torch.unique(batched_tokens)
new_toks = torch.zeros_like(batched_tokens)
for i, u in enumerate(unique):
    idx = torch.where(batched_tokens==u)
    new_toks[idx] =i
    

plt.imshow(new_toks.unsqueeze(0))
plt.axes('off')
plt.show()

