import sys
from dataclasses import dataclass
sys.path.append('../.')

from transformers import AutoTokenizer, AutoConfig
from rich.console import Console
from rich.text import Text
import torch
from utils import unique_third_order_markov_sequence, unique_second_order_markov_sequence


# model configs
model_name = 'Qwen/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
c = AutoConfig.from_pretrained(model_name)

# pretend we have argparse
@dataclass
class Args:
    n_reps: int = 3
    n_permute: int = 3
    chunk_size: int = 3
    n_permute_primitive: int = 3

# shared params
args = Args()
high_perm_colours = ["#AD8301", "#66800B", "#24837B"]
unique_colors = [ "#500DE0", "#9C370C", "#FFFFFF"]

for i in range(2):
    unique_tokens = torch.randint(0, c.vocab_size, (args.chunk_size,), generator=torch.manual_seed(i+123456))
    all_tokens, _, high_perms, low_perms  = unique_third_order_markov_sequence(unique_tokens, args, return_perms=True)

    _, high_perms_int = torch.unique(high_perms, dim=0, return_inverse=True)
    high_perms_int = high_perms_int.repeat_interleave(args.n_permute * args.n_permute_primitive)

    _, low_perms_int = torch.unique(low_perms, dim=0, return_inverse=True)
    low_perms_int = low_perms_int.repeat_interleave(args.n_permute_primitive)

    words = tokenizer.batch_decode(all_tokens)
    word_to_color_map = {j: unique_colors[j % len(unique_colors)] for j in range(low_perms_int.max()+1)}
    high_perm_to_color_map = {j: high_perm_colours[j % len(high_perm_colours)] for j in range(high_perms_int.max()+1)}

    console = Console(record=True)
    text = Text()
    word_counter = 0
    word_max = args.n_permute * args.n_permute_primitive
    for word_pos, word in enumerate(words):
        bg_color = high_perm_to_color_map[high_perms_int[word_pos].item()]
        color = word_to_color_map[low_perms_int[word_pos].item()]
        text.append(f"{word}", style=f"bold {color} on {bg_color}")
        word_counter += 1
        if word_counter == word_max:
            text.append("\n")
            word_counter = 0
        else:
            text.append(" | ", style=f"bold black on {bg_color}")

    console.print(text)
    console.save_html(f"figures/example3_{i}.html")

    all_tokens, _, perms = unique_second_order_markov_sequence(unique_tokens, args, return_perms=True)
    
    _, perms_int = torch.unique(perms, dim=0, return_inverse=True)
    perms_int = perms_int.repeat_interleave(args.n_permute)
    
    words = tokenizer.batch_decode(all_tokens)
    word_to_color_map = {j: high_perm_colours[j % len(high_perm_colours)] for j in range(perms_int.max()+1)}

    console = Console(record=True)
    text = Text()
    word_counter = 0
    word_max = args.n_permute * args.n_reps
    for word_pos, word in enumerate(words):
        color = word_to_color_map[perms_int[word_pos].item()]
        text.append(f"{word}", style=f"bold {color}")
        word_counter += 1
        if word_counter == word_max:
            text.append("\n")
            word_counter = 0
        else:
            text.append(" | ", style="bold black")

    console.print(text)
    console.save_html(f"figures/example2_{i}.html")