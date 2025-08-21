import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

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
    parser.add_argument('--total_batch_size', default=64, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--cmap', default='cividis', type=str)   
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
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

layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
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
        
    #for head in heads:

        
    # compute model accuracy
    logits = output['logits']
    logits.shape
    pred = logits.argmax(dim=-1)
    acc = (pred[:, :-1] == batched_tokens[:, 1:]).float()
    accuracies.append(acc)

all_chunk_ids= torch.cat(all_chunk_ids, dim=0)

accuracies = torch.cat(accuracies, dim=0).cpu().float()


save_dir = Path(f'data/learning_scores/{args.model_name.split("/")[-1]}')
save_dir.mkdir(parents=True, exist_ok=True)
for layer in list(layer_dict.keys()):
    heads = layer_dict[layer]
    for head in heads:
        address = f'{layer}-{head}'
        attn_matrices = torch.cat(attn_heads[address], dim=0)
        pooled = get_chunks(attn_matrices)
        pooled.shape
        
        head_accs = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps)
        for i in range(1, pooled.size(1)):
            row_ideal = all_chunk_ids[:, i, :i]
            row_model = pooled[:, i, :i]
            most_attn_idx = row_model.argmax(dim=1)
            score = row_ideal[torch.arange(args.total_batch_size), most_attn_idx]
            #print(score.shape, 'score!')
            
            head_accs[:, i] = score
            
        
        torch.save(head_accs,f'{save_dir}/{address}_accs.pt')
torch.save(accuracies, f'{save_dir}/model_accs.pt')


# head=0
# fig, ax =plt.subplots(1, 1, figsize=(8, 8))

# for layer in list(layer_dict.keys()):
#     heads = layer_dict[layer]
#     for head in heads:
#         address = f'{layer}-{head}'
#         attn_matrices = torch.cat(attn_heads[address], dim=0)
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
# plt.savefig('figures/icl_heads.png', dpi=200)
# plt.show()

# fig, ax =plt.subplots(1, 1, figsize=(8, 8))
# ax.plot(accuracies.mean(dim=0))
# plt.tick_params(labelsize=12)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.ylabel('Model Accuracy', size=14)
# plt.xlabel('Repetition', size=14)
# plt.ylim([0, 1.1])
# plt.savefig('figures/model_preds.png', dpi=200)
# plt.show()

# head=3
# attn_matrices = torch.cat(attn_heads[head], dim=0)
# attn_matrices.shape
# attn_matrices.shape
# fig, ax =plt.subplots(1, 2, figsize=(8, 8))

# ax[0].imshow(all_chunk_ids[1])

# ax[1].imshow(attn_matrices[1].cpu().float())
# plt.show()
# row_ideal
# row_model
# # plotting
# lines = torch.arange(0, num_tokens+1, int(num_tokens/num_total_reps))[1:]

# sublines = torch.arange(0, num_tokens+1, int(num_tokens/(num_total_reps*args.n_reps)))[1:]

# labels = [f'■' for i in range(args.n_permute*args.n_reps)]
# centers = sublines - (num_tokens/(num_total_reps*args.n_reps))/2
# #these are the heads we're looking at
# indices = [0, 3, 4, 5, 6]

# for idx in indices:
#     plot_head(output['attentions'][14], idx=idx, centers=centers, labels=labels, lines=lines, sublines=sublines)