import torch
from matplotlib import pyplot as plt
from huggingface_hub import login
login('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')
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from collections import defaultdict
import nnsight
from nnsight import LanguageModel
from pathlib import Path
def get_chunks(A):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, i*args.chunk_size:(i+1)*args.chunk_size, j*args.chunk_size:(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    return B

def plot_head(attn, idx, centers, labels, lines, sublines):
    attn = attn.cpu().float()
    fig, ax = plt.subplots(1, 2, figsize=(6,12))

    ax[0].set_title(f'Head {idx+1}', fontsize=14)


    for l in sublines:
        ax[0].axhline(y=l, color='white', linestyle='-', alpha=0.75, linewidth=1)
        ax[0].axvline(x=l, color='white', linestyle='-', alpha=0.75, linewidth=1)

    ax[0].imshow(attn[0, idx], cmap=args.cmap)

    ax[1].imshow(chunk_id, cmap=args.cmap)

    for l in range(chunk_id.size(0)+1):
        ax[1].axhline(y=l-0.5, color='white', linestyle='-', alpha=0.75, linewidth=1)
        ax[1].axvline(x=l-0.5, color='white', linestyle='-', alpha=0.75, linewidth=1)



    ax[1].set_title('Correct context', fontsize=14)
    ax[1].axis('off')
    ax[0].axis('off')
    plt.savefig(f'figures/shuffled/head_{idx}.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()




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

ablate_dict = {2:[3], 8:[3]}

layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
accuracies = []
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
            # compute model accuracy
        logits = output['logits']
        logits.shape
        pred = logits.argmax(dim=-1)
        acc = (pred[:, :-1] == batched_tokens[:, 1:]).float()
        accuracies.append(acc)
        for layer in list(layer_dict.keys()):
            heads = layer_dict[layer]
            for head in heads:
                address = f'{layer}-{head}'
                attn_heads[address].append(output['attentions'][layer][:, head])
            
                
all_chunk_ids= torch.cat(all_chunk_ids, dim=0)

save_dir = Path(f'data/learning_scores_ablated/{args.model_name.split("/")[-1]}')
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

