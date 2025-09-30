import warnings

warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import nnsight
import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    unique_second_order_markov_sequence,
    unique_third_order_markov_sequence,
)


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--n_reps', default=8, type=int)
    parser.add_argument('--nback', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=256, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--test_size', default=0.25, type=float)
    parser.add_argument('--module', default='heads', type=str, choices=['heads', 'mlp', 'attn', 'residual'])
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size // args.batch_size
    if args.markov_order==3:
        args.chunk_size = args.chunk_size//2
    return args

def get_chunks(A, args):
    """
    Pools a raw attention matrix into a chunk-by-chunk attention matrix.
    This function is identical to the one in find_learning_heads.py.
    """
    n_chunks = args.n_permute * args.n_reps
    B = torch.zeros(A.size(0), n_chunks, n_chunks, device=A.device)
    for i in range(n_chunks):
        for j in range(n_chunks):
            B[:, i, j] = A[:, i*args.chunk_size:(i+1)*args.chunk_size, j*args.chunk_size:(j+1)*args.chunk_size].mean(dim=(-1, -2))
    return B

def calculate_nback_identity(all_chunk_ids_for_batch, args):
    """
    Calculates the raw attention accuracy for a given head,
    replicating the logic from find_learning_heads.py.
    Returns both the overall score and the per-chunk accuracies.
    """
    
    # Step 2: Calculate accuracy based on the pooled matrix
    batch_size, n_chunks = all_chunk_ids_for_batch.size(0), all_chunk_ids_for_batch.size(1)
    #print(batch_size, all_chunk_ids_for_batch.shape)
    identities = torch.zeros(batch_size, n_chunks, device=all_chunk_ids_for_batch.device)

    for i in range(1, n_chunks):
        row_ideal = all_chunk_ids_for_batch[:, i, :i]
        
        if row_ideal.size(1) == 0:
            continue

        nback_idx = max(0, i-args.nback)
        batch_indices = torch.arange(batch_size, device=row_ideal.device)
        score = row_ideal[batch_indices, nback_idx]

        identities[:, i] = score
    
    return identities


def main():
    args = get_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nnsight.LanguageModel(args.model_name, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager")
    config = model.config
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // n_heads

    # --- Data Generation ---
    all_batched_tokens = []
    all_chunk_ids = []
    for _ in range(args.iters):
        batched_tokens = []
        chunk_ids = []
        for _ in range(args.batch_size):
            tokens = torch.randint(vocab_size, (args.chunk_size, ))
            if args.markov_order == 2:
                all_tokens, chunk_id = unique_second_order_markov_sequence(tokens, args)
                
            elif args.markov_order == 3:
                all_tokens, chunk_id = unique_third_order_markov_sequence(tokens, args)
                
            batched_tokens.append(all_tokens)
            chunk_ids.append(chunk_id)
            

        all_batched_tokens.append(torch.stack(batched_tokens))
        all_chunk_ids.append(torch.stack(chunk_ids))

    # --- Identify Targets ---
    if args.module == 'heads':
        targets = []
        for layer in range(config.num_hidden_layers):
            for head in range(n_heads):
                targets.append((layer, head))
        print(f"Starting analysis for all {len(targets)} heads.")
    else:
        targets = list(range(config.num_hidden_layers))
        print(f"Starting analysis for all {len(targets)} layers for module '{args.module}'.")

    # --- Feature and Label Extraction (Process in batches) ---
    head_activities = {target: [] for target in targets}
    head_labels = {target: [] for target in targets}

    
    # Process each batch separately to avoid memory issues
    for batch_idx in range(args.iters):
        print(f"Processing batch {batch_idx + 1}/{args.iters}...")
        
        batch_tokens = all_batched_tokens[batch_idx]
        batch_chunk_ids = all_chunk_ids[batch_idx]
        
        saved_activations = {}
        with torch.no_grad():
            with model.trace(batch_tokens, scan=False):
                # Pre-calculate which layers we need to save activations from
                unique_layers = sorted(list(set(t[0] if isinstance(t, tuple) else t for t in targets)))

                for layer in unique_layers:
                    if args.module == 'heads':
                        saved_activations[layer] = model.model.layers[layer].self_attn.o_proj.input.save()
                    elif args.module == 'mlp':
                        saved_activations[layer] = model.model.layers[layer].mlp.output.save()
                    elif args.module == 'attn':
                        saved_activations[layer] = model.model.layers[layer].self_attn.output[0].save()
                    elif args.module == 'residual':
                        saved_activations[layer] = model.model.layers[layer].output[0].save()
                
                
        # Update chunk_size for third-order Markov sequences
        current_chunk_size = args.chunk_size * args.n_permute_primitive if args.markov_order == 3 else args.chunk_size
        labels = calculate_nback_identity(all_chunk_ids_for_batch=batch_chunk_ids, args=args)
        labels = labels[:, 1:]
        labels = labels.flatten(0, 1)
        for target in targets:
            if args.module == 'heads':
                layer, head = target
                
                # Process activities
                o_proj_in_tensor = saved_activations[layer].value
                activity = rearrange(o_proj_in_tensor, 'b s (h d) -> b s h d', h=n_heads, d=head_dim)[:, :, head, :]
                
            else:
                layer = target
                activity = saved_activations[layer].value

            # --- POOLING AND LABEL GENERATION ---
            # Pool the activations to match the chunk-level labels
            n_chunks = args.n_permute * args.n_reps
            activity = activity.view(activity.size(0), n_chunks, current_chunk_size, -1).mean(dim=2)
            activity = activity[:, 1:]
            activity_for_classification = activity.flatten(0, 1)

            
            # Store the batch results
            head_activities[target].append(activity_for_classification.float().cpu())
            head_labels[target].append(labels.float().cpu())
        
        # Clear GPU memory after each batch
        del saved_activations, batch_tokens, batch_chunk_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all batch results
    print("Concatenating batch results...")
    for target in targets:
        head_activities[target] = torch.cat(head_activities[target], dim=0).numpy()
        head_labels[target] = torch.cat(head_labels[target], dim=0).numpy()

    # --- Classification ---
    results = defaultdict(list)
    if args.module=='heads':
        scores = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads)
    else:
        scores = torch.zeros(model.config.num_hidden_layers)
    for target in targets:
        if args.module == 'heads':
            layer, head = target
            address = f'{layer}-{head}'
        else:
            layer = target
            address = f'{layer}'

        X = head_activities[target] # (n_valid_chunks_total, activity_dim)
        y = head_labels[target]     # (n_valid_chunks_total,)

        # Use standard train-test split on the chunk-level data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > 0.65 and args.module == 'heads':
            results[layer].append(head)
        if args.module == 'heads':
            scores[layer, head] = acc
        else:
            scores[layer] = acc
        print(f"Module {args.module} {address}: Classification accuracy = {acc:.4f}, avg head acc: {y.mean().item()}, prediction_mean: {y_pred.mean()}")

    order = 'markov3' if args.markov_order==3 else 'markov2'
    model_str = args.model_name.split("/")[-1]
    save_dir = Path(f'data/one_back_scores/{order}/{model_str}/{args.module}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(scores, f'{save_dir}/decoding_accuracies.pt')
    torch.save(args, f'{save_dir}/args.pt')
    if args.module =='heads':
        torch.save(results, f'{save_dir}/one_back_heads.pt')
    print(f"Saved classification results to {save_dir} folder")

if __name__ == '__main__':
    main()
