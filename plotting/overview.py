
import sys
from dataclasses import dataclass
import sys
sys.path.insert(0, '/Users/tankredsaanum/Documents/smart_induction_heads')

#sys.path.append("..")

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import first_order_markov_sequence, unique_second_order_markov_sequence, unique_third_order_markov_sequence, get_chunk_ids_in_order, get_chunks

_ = torch.set_grad_enabled(False)
model_name = "Qwen/Qwen2.5-1.5B"
mpl.rcParams['mathtext.fontset'] = 'cm'

torch.random.manual_seed(42)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_chunks(A, args):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            B[:, i, j] = A[:, (i*args.chunk_size):(i+1)*args.chunk_size, (j*args.chunk_size):(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    return B

def get_chunks_3rd_order(A, args):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps, args.n_permute*args.n_reps)
    higher_order_chunk_size = args.chunk_size * args.n_permute_primitive
    for i in range(args.n_permute*args.n_reps):
        for j in range(args.n_permute*args.n_reps):
            rows = A[:, (i*higher_order_chunk_size):(i+1)*higher_order_chunk_size, :]
            transition_idx = torch.arange(1, higher_order_chunk_size+1)
            mask = transition_idx % (higher_order_chunk_size//args.n_permute_primitive) == 0
            mask[-1] = False
            rows = rows[:, mask]
            patch_score = rows[:, :, (j*higher_order_chunk_size):(j+1)*higher_order_chunk_size]
            
            B[:, i, j] = patch_score.reshape(args.total_batch_size, -1).mean(dim=-1)
            
    return B

def get_chunks_3rd_order_primitive(A, args):
    B = torch.zeros(args.total_batch_size, args.n_permute*args.n_reps*args.n_permute_primitive, args.n_permute*args.n_reps*args.n_permute_primitive)
    for i in range(args.n_permute*args.n_reps*args.n_permute_primitive):
        for j in range(args.n_permute*args.n_reps*args.n_permute_primitive):
            B[:, i, j] = A[:, (i*args.chunk_size):(i+1)*args.chunk_size, (j*args.chunk_size):(j+1)*args.chunk_size].reshape(args.total_batch_size, -1).mean(dim=-1)
    return B

@dataclass
class Args:
    total_batch_size=8
    chunk_size: int = 3
    n_reps: int = 5
    n_permute: int = 3
    n_permute_primitive: int = 3

basic_args = Args(n_reps=2)
args = Args()
third_order_args = Args(n_reps=5)


#tokens = torch.randint(low=0, high=150000, size=(args.chunk_size,))
#tokens = torch.randperm(150000)[:args.chunk_size]
tokens = torch.tensor([0, 1, 2])
seq = first_order_markov_sequence(tokens, basic_args)
seq2, chunk2, perms2 = unique_second_order_markov_sequence(tokens, args, return_perms=True)
seq3, chunk3, perms3, primitveperms3  = unique_third_order_markov_sequence(tokens, third_order_args, return_perms=True)


seq = seq.unsqueeze(0).repeat(basic_args.total_batch_size, 1)
seq2 = seq2.unsqueeze(0).repeat(args.total_batch_size, 1)
seq3 = seq3.unsqueeze(0).repeat(third_order_args.total_batch_size, 1)
for i in range(args.total_batch_size):
    mapping = torch.randperm(150000)[:args.chunk_size]
    for j in range(args.chunk_size):
        new_token = mapping[j]
        seq_1idx = torch.where(seq[i] == j)[0]
        seq[i, seq_1idx] = new_token
        
        seq_2idx = torch.where(seq2[i] == j)[0]
        seq2[i, seq_2idx] = new_token
        
        seq_3idx = torch.where(seq3[i] == j)[0]
        seq3[i, seq_3idx] = new_token


learning_scores = torch.load("data/learning_scores/markov3/Qwen2.5-1.5B/learning_scores.pt")


k = 1 # The rank of the layer/head pair to select (e.g., 10 for the 10th best)

# Flatten the learning_scores tensor to find top k scores across all layers and heads
flat_scores = learning_scores.flatten()

# Get the top k scores and their flat indices
# We need to find at least k top scores to select the k-th one.
top_k_scores, top_k_flat_indices = torch.topk(flat_scores, k)

# Convert the flat indices to 2D indices (layer, head)
top_k_indices = [torch.unravel_index(i, learning_scores.shape) for i in top_k_flat_indices]

# Select the k-th best layer/head pair (k-1 because of 0-indexing)
chosen_layer, chosen_head = top_k_indices[k-1]
chosen_score = top_k_scores[k-1]

print(f"Selected {k}-th top layer/head: ({chosen_layer.item()}, {chosen_head.item()}) with score {chosen_score.item():.4f}")
print(f"\nTop {k} layer/head pairs and their scores:")
for i in range(k):
    layer, head = top_k_indices[i]
    score = top_k_scores[i]
    print(f"  {i+1}. Layer {layer.item()}, Head {head.item()}: {score.item():.4f}")


seq_output = model(seq.to(model.device),output_attentions=True).attentions
seq_2_output = model(seq2.to(model.device),output_attentions=True).attentions
seq_2_chunks = get_chunks(seq_2_output[chosen_layer][:, chosen_head], args=args)
seq_3_output = model(seq3.to(model.device),output_attentions=True).attentions
seq_3_chunks = get_chunks_3rd_order(seq_3_output[chosen_layer][:, chosen_head], args=third_order_args)
seq_3_chunks_primitive = get_chunks_3rd_order_primitive(seq_3_output[chosen_layer][:, chosen_head], args=third_order_args)

seq_2_chunks = seq_2_chunks.mean(dim=0)
seq_3_chunks = seq_3_chunks.mean(dim=0)
seq_3_chunks_primitive = seq_3_chunks_primitive.mean(dim=0)


seq = seq[0]
seq2 = seq2[0]
seq3 = seq3[0]

def add_attention_borders(ax, attention_matrix, token_sequence):
    """
    Add colored borders to highlight a context-aware copying pattern.

    For each token in the sequence (a query at position `i`), this function:
    1. Identifies the actual next token in the sequence (at `i+1`), which is the prediction target.
    2. Finds the position `max_col` where the query token `i` pays the most attention.
    3. Checks if the token at `max_col` is the same as the target token from step 1.
    4. If they match, the border is green (the head is attending to a previous
       instance of the correct next token).
    5. Otherwise, the border is red.
    """
    seq_len = attention_matrix.shape[0]
    # Loop up to the second-to-last token, as the last token has no "next" token.
    for row in range(1, seq_len - 1):
        # The token that should be predicted next.
        target_token = token_sequence[row + 1]

        # Find the column with maximum attention for the current row (query).
        # We only look at positions up to the current one.
        max_col = torch.argmax(attention_matrix[row, :row + 1]).item()

        # The token that is actually being attended to.
        attended_token = token_sequence[max_col]

        # Check if the head is attending to a previous instance of the correct target token.
        if attended_token == target_token:
            border_color = 'green'
        else:
            border_color = 'red'

        # Add a rectangle border around the cell with the highest attention.
        rect = plt.Rectangle((max_col, row), 1, 1,
                           fill=True, edgecolor=border_color, facecolor=border_color,
                           linewidth=1, zorder=10, clip_on=False, alpha=0.5)
        ax.add_patch(rect)


def add_copying_attention_borders(ax, attention_matrix, token_sequence):
    """
    Add colored borders to cells with maximum attention per row,
    highlighting patterns of attending to previous instances of the same token.

    For each token in the sequence (a query at position `row`), this function:
    1. Finds the position of the token with the highest attention score (`max_col`).
    2. Checks if the token at `max_col` is the same as the current token.
    3. If it is, the border is green (indicating attention to a previous instance of itself).
    4. Otherwise, the border is red.
    """
    seq_len = attention_matrix.shape[0]
    for row in range(1, seq_len):  # Start from 1, as token 0 has no history
        current_token = token_sequence[row]
        
        # Find the column with maximum attention for this row, up to the current position
        max_col = torch.argmax(attention_matrix[row, :row]).item()

        # Get the token ID at the position of maximum attention
        attended_token = token_sequence[max_col]

        # Check if the head is attending to a previous instance of the same token
        if attended_token == current_token:
            border_color = 'green'
        else:
            border_color = 'red'

        # Add a rectangle border around the cell with the highest attention
        # specify both edgecolor and the filling color
        rect = plt.Rectangle((max_col, row), 1, 1,
                           fill=True, edgecolor=border_color, facecolor=border_color,
                           linewidth=1, zorder=10, clip_on=False, alpha=0.5)
        ax.add_patch(rect)

large_fs=50
fig, axs = plt.subplots(3, 3, figsize=(30, 30))
for ax in axs.flatten():
    ax.tick_params(axis=u'both', which=u'both',length=0)
for i in range(3):
    for j in range(3):
        if i < j:
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_xlabel("")
            axs[i, j].set_ylabel("")
            for spine in axs[i, j].spines.values():
                spine.set_visible(False)

unique_tokens = list(set(seq.tolist()))
unique_tokens.sort()

start_chr = 97
token_to_letter = {token: chr(start_chr+i) for i, token in enumerate(unique_tokens)}
seq_letters = [token_to_letter[token] for token in seq.tolist()]
attention_map1 = seq_output[chosen_layer][:, chosen_head].cpu().float().mean(dim=0)

sns.heatmap(attention_map1,cbar=False, cmap="copper",linewidths=1, linecolor='black', ax=axs[0,0])
add_attention_borders(axs[0,0], attention_map1, seq)
axs[0,0].set_xticks(torch.arange(len(seq_letters))+0.5)
axs[0,0].set_yticks(torch.arange(len(seq_letters))+0.5)
axs[0,0].set_xticklabels([f"${letter}$" for letter in seq_letters], fontsize=large_fs-5)
axs[0,0].set_yticklabels([f"${letter}$" for letter in seq_letters], fontsize=large_fs-5, rotation=0)

sns.heatmap(seq_2_chunks.cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[1,0])
chunk_ids = get_chunk_ids_in_order(perms2)
add_copying_attention_borders(axs[1,0], seq_2_chunks.cpu().float(), torch.tensor(chunk_ids))
id_to_letter = {i: chr(945 + i) for i in range(len(set(chunk_ids)))}
axs[1,0].set_xticks(torch.arange(len(chunk_ids)) + 0.5)
axs[1,0].set_yticks(torch.arange(len(chunk_ids)) + 0.5)
axs[1,0].set_xticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=large_fs)
axs[1,0].set_yticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=large_fs, rotation=0)

unique_tokens = list(set(seq2.tolist()))
unique_tokens.sort()
token_to_letter = {token: chr(start_chr + i) for i, token in enumerate(unique_tokens)}
seq2_letters = [token_to_letter[token] for token in seq2.tolist()]

attention_map2 = seq_2_output[chosen_layer][:, chosen_head].cpu().float().mean(dim=0)
sns.heatmap(attention_map2, cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[1,1])
add_attention_borders(axs[1,1], attention_map2, seq2)
axs[1,1].set_xticks(torch.arange(len(seq2_letters)) + 0.5)
axs[1,1].set_yticks(torch.arange(len(seq2_letters)) + 0.5)
axs[1,1].set_xticklabels([f"${letter}$" for letter in seq2_letters], fontsize=15, rotation=0)
axs[1,1].set_yticklabels([f"${letter}$" for letter in seq2_letters], fontsize=15, rotation=0)


sns.heatmap(seq_3_chunks.cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[2,0])
chunk_ids = get_chunk_ids_in_order(perms3)
# use uppercase phi, psi, and omega for the top level chunk ids
# make sure we use the uppercase letters in latex format

greek_letters = ['\u03A9', '\u03C8', '\u03C6'] # Ω, ψ, φ
id_to_letter = {i: greek_letters[i] for i in range(len(set(chunk_ids)))}
add_copying_attention_borders(axs[2,0], seq_3_chunks.cpu().float(), torch.tensor(chunk_ids))
axs[2,0].set_xticks(torch.arange(len(chunk_ids)) + 0.5)
axs[2,0].set_yticks(torch.arange(len(chunk_ids)) + 0.5)
axs[2,0].set_xticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=large_fs)
axs[2,0].set_yticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=large_fs, rotation=0)
# Second subplot for the chunked attention (averaged)
sns.heatmap(seq_3_chunks_primitive.cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[2,1])
chunk_ids = get_chunk_ids_in_order(primitveperms3)
add_copying_attention_borders(axs[2,1], seq_3_chunks_primitive.cpu().float(), torch.tensor(chunk_ids))
id_to_letter = {i: chr(945 + i) for i in range(len(set(chunk_ids)))}
axs[2,1].set_xticks(torch.arange(len(chunk_ids)) + 0.5)
axs[2,1].set_yticks(torch.arange(len(chunk_ids)) + 0.5)
axs[2,1].set_xticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=15, rotation=0)
axs[2,1].set_yticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=15, rotation=0)
# Third subplot for the full attention matrix
unique_tokens = list(set(seq3.tolist()))
unique_tokens.sort()
token_to_letter = {token: chr(start_chr+i) for i, token in enumerate(unique_tokens)}
seq3_letters = [token_to_letter[token] for token in seq3.tolist()]

attention_map3 = seq_3_output[chosen_layer][:, chosen_head].cpu().float().mean(dim=0)
sns.heatmap(attention_map3, cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[2,2])
add_attention_borders(axs[2,2], attention_map3, seq3)

axs[2,2].set_xticks(torch.arange(len(seq3_letters)) + 0.5)
axs[2,2].set_yticks(torch.arange(len(seq3_letters)) + 0.5)
axs[2,2].set_xticklabels([f"${letter}$" for letter in seq3_letters], fontsize=8, rotation=0)
axs[2,2].set_yticklabels([f"${letter}$" for letter in seq3_letters], fontsize=8, rotation=0)

lwd=1.0
for i in range(1, third_order_args.n_reps * third_order_args.n_permute * third_order_args.n_permute_primitive):
    axs[2,2].axhline(i * third_order_args.chunk_size * third_order_args.n_permute, color='white', linewidth=lwd, zorder=10)
    axs[2,2].axvline(i * third_order_args.chunk_size* third_order_args.n_permute , color='white', linewidth=lwd, zorder=10)
for i in range(1, third_order_args.n_reps * third_order_args.n_permute):
    axs[2,1].axhline(i * third_order_args.chunk_size, color='white', linewidth=lwd, zorder=10)
    axs[2,1].axvline(i * third_order_args.chunk_size, color='white', linewidth=lwd, zorder=10)
    
for i in range(1, args.n_reps * args.n_permute):
    axs[1,1].axhline(i * args.chunk_size, color='white', linewidth=lwd, zorder=10)
    axs[1,1].axvline(i * args.chunk_size, color='white', linewidth=lwd, zorder=10)

# save as svg, png, and pdf
plt.tight_layout()
plt.savefig("figures/overview.svg", format="svg")
plt.savefig("figures/overview.png", format="png")
plt.savefig("figures/overview.pdf", format="pdf")
plt.show()


