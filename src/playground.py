from dataclasses import dataclass

from fastcore.script import call_parse
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from utils import unique_second_order_markov_sequence, unique_third_order_markov_sequence, unique_fourth_order_markov_sequence, unique_fifth_order_markov_sequence

_ = torch.set_grad_enabled(False)

@dataclass
class Args:
    chunk_size: int = 4
    n_reps: int = 4
    n_permute: int = 4
    n_permute_primitive: int = 4
    markov_order: int = 5
    repeats: int = 5

@call_parse
def main(
    model_name:str = "Qwen/Qwen2.5-1.5B", 
    chunk_size:int=4, 
    n_reps:int=4, 
    n_permute:int=4, 
    n_permute_primitive:int=4, 
    order:int=5,
    markov_order: int = 5,
    repeats: int = 5,
):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval()
    args = Args(
        chunk_size=chunk_size, 
        n_reps=n_reps, 
        n_permute=n_permute, 
        n_permute_primitive=n_permute_primitive,
        markov_order=order,
        repeats=repeats,
        markov_order=markov_order,
    )

    all_accs = []
    for repeat in tqdm(range(args.repeats)):
        tokens = torch.randint(model.config.vocab_size, size=(args.chunk_size,))

        if args.markov_order == 2:
            seq, _, _ = unique_second_order_markov_sequence(tokens, args, return_perms=True)
    
        elif args.markov_order == 3:
            seq, _, _, _ = unique_third_order_markov_sequence(tokens, args, return_perms=True)
            l2_chunk_size = args.chunk_size * args.n_permute_primitive
    
        elif args.markov_order == 4:
            seq, _, _ = unique_fourth_order_markov_sequence(tokens, args, return_perms=True)
            l3_chunk_size = args.chunk_size * args.n_permute_primitive * args.n_permute_primitive
    
        elif args.markov_order == 5:
            seq, _, _ = unique_fifth_order_markov_sequence(tokens, args, return_perms=True)
            l4_chunk_size = args.chunk_size * args.n_permute_primitive * args.n_permute_primitive * args.n_permute_primitive


        seq = seq.unsqueeze(0).to(model.device)
        logits = model(seq).logits
        pred = logits.argmax(dim=-1)
        accs = (pred[:, :-1] == seq[:, 1:]).float()

        if args.markov_order == 2:
            accs = accs.view(accs.size(0), -1, args.chunk_size)
            accs = accs.mean(dim=0)
            accs = accs[:, :-1].mean(dim=1)
        elif args.markov_order == 3:
            accs = accs.view(accs.size(0), -1, l2_chunk_size)
            accs = accs.mean(dim=0)
            mask = torch.arange(1, l2_chunk_size + 1) % (l2_chunk_size // args.n_permute_primitive) == 0
            mask[-1] = False
            accs = accs[:, mask]
            accs = accs.mean(dim=1)
        elif args.markov_order == 4:
            accs = accs.view(accs.size(0), -1, l3_chunk_size)
            accs = accs.mean(dim=0)
            l2_chunk_size_inner = l3_chunk_size // args.n_permute
            mask = torch.arange(1, l3_chunk_size + 1) % l2_chunk_size_inner == 0
            mask[-1] = False
            accs = accs[:, mask]
            accs = accs.mean(dim=1)
        elif args.markov_order == 5:
            # Reshape to view the sequence as a series of L4 chunks
            num_l4_chunks = accs.shape[1] // l4_chunk_size
            accs_chunked = accs.view(accs.size(0), num_l4_chunks, l4_chunk_size)
            # Select accuracy at the first position of each L4 chunk (after the first)
            # This tests the model's ability to predict across the L4 boundary.
            transition_accs = accs_chunked[:, 1:, 0]
            accs = transition_accs.mean(dim=1)


        all_accs.append(accs.mean().item())

    torch.save(torch.tensor(all_accs), f'order_{args.markov_order}_{model_name.split("/")[-1]}.pt')

    print(f"Mean accuracy for order {args.markov_order}: {torch.tensor(all_accs).mean().item()}")
