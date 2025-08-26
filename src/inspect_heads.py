import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

def get_chunks(A):
    B = torch.zeros(A.size(0), args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, i*args.chunk_size:(i+1)*args.chunk_size, j*args.chunk_size:(j+1)*args.chunk_size].reshape(1, -1).mean(dim=-1)
    return B



def get_config():
    parser = ArgumentParser()

    parser.add_argument('--n_pad_repetitions', default=4, type=int)
    parser.add_argument('--n_reps', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=8, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--cmap', default='cividis', type=str)   
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)   
    args, _ = parser.parse_known_args()

    return args
args = get_config()
args.iters = args.total_batch_size//args.batch_size
device='mps'
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto")
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config

attn_heads = defaultdict(list)
all_chunk_ids =[]
accuracies = []

#layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(n_heads))

for iter in range(args.iters):
    print(iter/args.iters)
    batched_tokens = []
    chunk_ids = []

    for _ in range(args.batch_size):
        tokens = torch.randint(vocab_size, (args.chunk_size, ))

        perms = []
        for _ in range(args.n_permute):
            perm_idx = torch.randperm(args.chunk_size)
            perms.append(tokens[perm_idx])

        ordered_sequence = torch.arange(args.n_reps*args.n_permute)%args.n_permute
        permuted_sequence = ordered_sequence[torch.randperm(args.n_reps*args.n_permute)]
        all_tokens = []
        for seq_id in permuted_sequence:
            all_tokens.append(perms[seq_id])

        all_tokens = torch.cat(all_tokens, dim=0)
        batched_tokens.append(all_tokens)
        # let's get a matrix showing which token chunks are identical (e.g. 0 L0 distance)
        chunk_id=(torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
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
            

if True:
    attn = attn_heads['15-1'][0][0].cpu().float()
    pooled = get_chunks(attn.unsqueeze(0)).squeeze(0)
    f, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].imshow(pooled)
    ax[1].imshow(all_chunk_ids[0][0])
    plt.show()
    for i in range(pooled.size(0)):
        row = pooled[i, :i]
        print(i, row, pooled[i])
    
pooled.shape