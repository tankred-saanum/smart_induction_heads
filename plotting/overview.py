
import sys
from dataclasses import dataclass
sys.path.append("..")

import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import first_order_markov_sequence, unique_second_order_markov_sequence, unique_third_order_markov_sequence, get_chunk_ids_in_order, get_chunks, get_chunks_3rd_order

_ = torch.set_grad_enabled(False)
model_name = "Qwen/Qwen2.5-1.5B"
mpl.rcParams['mathtext.fontset'] = 'cm'

torch.random.manual_seed(1234)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(model_name)


@dataclass
class Args:
    chunk_size: int = 3
    n_reps: int = 5
    n_permute: int = 3
    n_permute_primitive: int = 3

basic_args = Args(n_reps=3)
args = Args()
third_order_args = Args(n_reps=8)


tokens = torch.randint(low=100, high=1000, size=(args.chunk_size,))


seq = first_order_markov_sequence(tokens, basic_args)
seq2, chunk2, perms2 = unique_second_order_markov_sequence(tokens, args, return_perms=True)
seq3, chunk3, perms3, primitveperms3  = unique_third_order_markov_sequence(tokens, third_order_args, return_perms=True)


learning_scores = torch.load("../data/learning_scores/markov3/Qwen2.5-1.5B/learning_scores.pt")


k = 5 # The rank of the layer/head pair to select (e.g., 10 for the 10th best)

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



seq_output = model(seq.unsqueeze(0).to(model.device),output_attentions=True).attentions
seq_2_output = model(seq2.unsqueeze(0).to(model.device),output_attentions=True).attentions
seq_2_chunks = get_chunks(seq_2_output[chosen_layer][0][chosen_head], args)
seq_3_output = model(seq3.unsqueeze(0).to(model.device),output_attentions=True).attentions
seq_3_chunks, seq_3_chunks_primitive = get_chunks_3rd_order(seq_3_output[chosen_layer][0][chosen_head], third_order_args)


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
token_to_letter = {token: chr(97+i) for i, token in enumerate(unique_tokens)}
seq_letters = [token_to_letter[token] for token in seq.tolist()]
sns.heatmap(seq_output[chosen_layer][0][chosen_head].cpu().float(),cbar=False, cmap="copper",linewidths=1, linecolor='black', ax=axs[0,0])
add_attention_borders(axs[0,0], seq_output[chosen_layer][0][chosen_head].cpu().float(), seq)
axs[0,0].set_xticks(torch.arange(len(seq_letters))+0.5)
axs[0,0].set_yticks(torch.arange(len(seq_letters))+0.5)
axs[0,0].set_xticklabels([f"${letter}$" for letter in seq_letters], fontsize=30)
axs[0,0].set_yticklabels([f"${letter}$" for letter in seq_letters], fontsize=30, rotation=0)


sns.heatmap(seq_2_chunks.cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[1,0])
chunk_ids = get_chunk_ids_in_order(perms2)
add_copying_attention_borders(axs[1,0], seq_2_chunks.cpu().float(), torch.tensor(chunk_ids))
id_to_letter = {i: chr(945 + i) for i in range(len(set(chunk_ids)))}
axs[1,0].set_xticks(torch.arange(len(chunk_ids)) + 0.5)
axs[1,0].set_yticks(torch.arange(len(chunk_ids)) + 0.5)
axs[1,0].set_xticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=30)
axs[1,0].set_yticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=30, rotation=0)

unique_tokens = list(set(seq2.tolist()))
unique_tokens.sort()
token_to_letter = {token: chr(97 + i) for i, token in enumerate(unique_tokens)}
seq2_letters = [token_to_letter[token] for token in seq2.tolist()]

sns.heatmap(seq_2_output[chosen_layer][0][chosen_head].cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[1,1])
add_attention_borders(axs[1,1], seq_2_output[chosen_layer][0][chosen_head].cpu().float(), seq2)
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
axs[2,0].set_xticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=30)
axs[2,0].set_yticklabels([f"${id_to_letter[id]}$" for id in chunk_ids], fontsize=30, rotation=0)
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
token_to_letter = {token: chr(97+i) for i, token in enumerate(unique_tokens)}
seq3_letters = [token_to_letter[token] for token in seq3.tolist()] 
sns.heatmap(seq_3_output[chosen_layer][0][chosen_head].cpu().float(), cbar=False, cmap="copper", linewidths=1, linecolor='black', ax=axs[2,2])
add_attention_borders(axs[2,2], seq_3_output[chosen_layer][0][chosen_head].cpu().float(), seq3)

axs[2,2].set_xticks(torch.arange(len(seq3_letters)) + 0.5)
axs[2,2].set_yticks(torch.arange(len(seq3_letters)) + 0.5)
axs[2,2].set_xticklabels([f"${letter}$" for letter in seq3_letters], fontsize=8, rotation=0)
axs[2,2].set_yticklabels([f"${letter}$" for letter in seq3_letters], fontsize=8, rotation=0)
# save as svg, png, and pdf
plt.tight_layout()
plt.savefig("../figures/overview.svg", format="svg")
plt.savefig("../figures/overview.png", format="png")
plt.savefig("../figures/overview.pdf", format="pdf")
plt.show()


