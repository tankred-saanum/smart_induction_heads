import sys
sys.path.insert(0, '/Users/tankredsaanum/Documents/smart_induction_heads')
#sys.path.insert(0, '/raven/u/tsaanum/modular')
sys.path.insert(0, '/n/home04/tsaanum/smart_induction_heads')
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from src.utils import first_order_markov_sequence, second_order_markov_sequence, third_order_markov_sequence, unique_third_order_markov_sequence, unique_second_order_markov_sequence
def get_chunks(A):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, i*args.chunk_size:(i+1)*args.chunk_size, j*args.chunk_size:(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    return B



def get_config():
    parser = ArgumentParser()

    parser.add_argument('--n_pad_repetitions', default=4, type=int)
    parser.add_argument('--n_reps', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=16, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--chunk_size', default=4, type=int)
    parser.add_argument('--markov_order', default=3, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--cmap', default='cividis', type=str)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)   
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size//args.batch_size

    return args
args = get_config()

device='mps'
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="auto")
config = PretrainedConfig.from_pretrained(args.model_name)
vocab_size = config.vocab_size
n_heads = config.num_attention_heads # number of heads in the models, should get info directly from config

accuracies = []


for iter in range(args.iters):
    print(iter/args.iters)
    batched_tokens = []

    for _ in range(args.batch_size):
        tokens = torch.randint(vocab_size, (args.chunk_size, ))
        if args.markov_order == 1:
            all_tokens = first_order_markov_sequence(tokens, args)
            
        elif args.markov_order == 2:
            all_tokens, _ = unique_second_order_markov_sequence(tokens, args)
            
        elif args.markov_order == 3:
            all_token, _ = unique_third_order_markov_sequence(tokens, args)
            
        batched_tokens.append(all_tokens)

    batched_tokens = torch.stack(batched_tokens, dim=0).to(device)

    with torch.no_grad():
        output = model(batched_tokens)

    logits = output['logits']
    pred = logits.argmax(dim=-1)
    acc = (pred[:, :-1] == batched_tokens[:, 1:]).float()
    accuracies.append(acc)

f, ax = plt.subplots(1, 2, figsize=(10, 8))
accuracies = torch.cat(accuracies, dim=0).cpu().float()


ax[0].plot(accuracies.mean(dim=0))

if args.markov_order == 1:
    A = accuracies.mean(dim=0)
elif args.markov_order == 2:
    A = torch.cat([accuracies, torch.ones(accuracies.size(0), 1)], dim=-1).view(accuracies.size(0), -1, args.chunk_size)
    A = A[:, :, :args.chunk_size-1].mean(dim=-1).mean(dim=0)

else:

    A = torch.cat([accuracies, torch.ones(accuracies.size(0), 1)], dim=-1).view(accuracies.size(0), args.n_permute, args.n_reps, args.chunk_size*args.n_permute_primitive).mean(dim=0)   
    A = A[:, :, :-1].mean(dim=-1).ravel()

ax[1].plot(A)
ax[1].set_ylim([0, 1.1])
ax[0].set_ylim([0, 1.1])
plt.show()

