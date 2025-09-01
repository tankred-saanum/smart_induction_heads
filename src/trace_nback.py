import warnings
warnings.filterwarnings("ignore")
from transformers import PretrainedConfig
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from einops import rearrange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import nnsight
import numpy as np
from utils import first_order_markov_sequence, second_order_markov_sequence, third_order_markov_sequence

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--n_reps', default=10, type=int)
    parser.add_argument('--nback', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=32, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test_size', default=0.25, type=float)
    parser.add_argument('--module', default='mlp', type=str, choices=['heads', 'mlp', 'attn', 'residual'])
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size // args.batch_size
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

def calculate_raw_attention_accuracy(all_batched_tokens, attn_for_head, all_chunk_ids_for_batch, args):
    """
    Calculates the raw attention accuracy for a given head,
    replicating the logic from find_learning_heads.py.
    Returns both the overall score and the per-chunk accuracies.
    """
    # Step 1: Pool the attention matrix into chunks, same as in find_learning_heads.py
    pooled_attn = get_chunks(attn_for_head, args)
    
    # Step 2: Calculate accuracy based on the pooled matrix
    batch_size, n_chunks = pooled_attn.size(0), pooled_attn.size(1)
    #print(batch_size, all_chunk_ids_for_batch.shape)
    head_accs = torch.zeros(batch_size, n_chunks, device=pooled_attn.device)

    for i in range(1, n_chunks):
        row_ideal = all_chunk_ids_for_batch[:, i, :i]
        row_model = pooled_attn[:, i, :i]
        
        if row_model.size(1) == 0:
            continue

        nback_idx = max(0, i-args.nback)
        most_attn_idx = row_model.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=row_ideal.device)
        score = row_ideal[batch_indices, nback_idx]
        #predicted_tokens = all_batched_tokens[batch_indices, most_attn_idx.to(row_ideal.device)]
        #targets = all_batched_tokens[batch_indices, i]
        #print(predicted_tokens.shape, targets.shape)
        #accs = predicted_tokens == targets
        #print(accs)
        #print(predicted_tokens.shape, accs.shape)
        #print(score.shape, batch_indices.shape, most_attn_idx.shape, row_model.shape, row_ideal.shape)
        head_accs[:, i] = score
    
    # Replicate the final score calculation from find_learning_heads.py
    #head_accs = head_accs.flatten(1, 2)
    #head_accs = head_accs[:, :, 0]
    if n_chunks > 1:
        half = head_accs.size(1)//2
        learning_score = head_accs.mean(dim=0)[half:].mean()
    else:
        learning_score = torch.tensor(0.0, device=head_accs.device)
        
    return learning_score.item(), head_accs


def main():
    args = get_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nnsight.LanguageModel(args.model_name, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager")
    config = model.config
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    head_dim = config.hidden_size // n_heads

    # --- Data Generation ---
    all_batched_tokens = []
    all_chunk_ids = []
    for _ in range(args.iters):
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
            
            # tokens = torch.randint(vocab_size, (args.chunk_size,))
            # perms = [tokens[torch.randperm(args.chunk_size)] for _ in range(args.n_permute)]
            # ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
            # permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
            # all_tokens = torch.cat([perms[seq_id] for seq_id in permuted_sequence], dim=0)
            # batched_tokens.append(all_tokens)
            # chunk_id = (torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
            # chunk_ids.append(chunk_id)
        all_batched_tokens.append(torch.stack(batched_tokens))
        all_chunk_ids.append(torch.stack(chunk_ids))

    all_batched_tokens = torch.cat(all_batched_tokens, dim=0)
    all_chunk_ids = torch.cat(all_chunk_ids, dim=0)

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

    # --- Feature and Label Extraction ---
    head_activities = {}
    head_labels = {}

    saved_activations = {}
    with torch.no_grad():
        with model.trace(all_batched_tokens, output_attentions=True, scan=False):
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
                    saved_activations[layer] = model.model.layers[layer].output.save()
            
            output = model.output.save()

    args.chunk_size = args.chunk_size * args.n_permute_primitive if args.markov_order == 3 else args.chunk_size
    # --- Identify Best Heads (if not in 'heads' mode) ---
    best_heads = {}
    best_head_accs = {}
    if args.module != 'heads':
        print("Calculating raw attention accuracies to find best head for each layer...")
        attentions_from_output = output['attentions']

        for layer in range(config.num_hidden_layers):
            head_accuracies = []
            head_acc_matrices = []
            for head in range(n_heads):
                attn_for_head = attentions_from_output[layer][:, head, :, :]
                raw_acc, head_acc_matrix = calculate_raw_attention_accuracy(all_batched_tokens,attn_for_head, all_chunk_ids, args)
                head_accuracies.append(raw_acc)
                head_acc_matrices.append(head_acc_matrix)
            
            if not head_accuracies:
                best_acc, best_head = 0.0, 0
                best_head_acc_matrix = torch.zeros_like(head_acc_matrices[0]) if head_acc_matrices else torch.zeros(args.total_batch_size, args.n_permute * args.n_reps)
            else:
                #best_acc, best_head = max(head_accuracies)
                best_head = torch.argmax(torch.tensor(head_accuracies))
                best_acc = head_accuracies[best_head]
                
                best_head_acc_matrix = head_acc_matrices[best_head]

            best_heads[layer] = best_head
            best_head_accs[layer] = best_head_acc_matrix
            print(f"Layer {layer}: Best head is {best_head} with raw attention accuracy {best_acc:.4f}")

    # Post-process to get labels and format data
    attentions_from_output = output['attentions']
    for target in targets:
        if args.module == 'heads':
            layer, head = target
            address = f'{layer}-{head}'
            
            # Process activities
            o_proj_in_tensor = saved_activations[layer].value
            activity = rearrange(o_proj_in_tensor, 'b s (h d) -> b s h d', h=n_heads, d=head_dim)[:, :, head, :]
            
            # Process attentions to get labels using this specific head
            attn_tensor = attentions_from_output[layer]
            attn_for_head = attn_tensor[:, head, :, :]
            
            # Calculate binary labels based on this head's attention accuracy
            _, head_acc_matrix = calculate_raw_attention_accuracy(all_batched_tokens,attn_for_head, all_chunk_ids, args)
            
        else:
            layer = target
            address = f'{layer}'
            activity = saved_activations[layer].value

            # Use the best head's accuracy matrix as binary labels
            head_acc_matrix = best_head_accs[layer]
        
        # --- POOLING AND LABEL GENERATION ---
        # Pool the activations to match the chunk-level labels
        n_chunks = args.n_permute * args.n_reps
        #print(activity.shape, head_acc_matrix.shape)
        activity = activity.view(activity.size(0),n_chunks, args.chunk_size, -1).mean(dim=2)
        activity = activity[:, 1:]
        activity_for_classification = activity.flatten(0, 1)
        labels = head_acc_matrix[:, 1:]
        labels = labels.flatten(0, 1)
        #print(activity_for_classification.shape, labels.shape)
       
        # print(activity_for_classification[:30, :5])
        # print(labels[:30])
        
        # pooled_activity = rearrange(activity, 'b (c s) d -> b c s d', c=n_chunks, s=args.chunk_size)
        # pooled_activity = pooled_activity.mean(dim=2) # Pool over chunk dimension

        # # Generate binary labels: 1 if correct, 0 if incorrect
        # # We use chunks 1 onwards (skip the first chunk which has no previous context)
        # labels = head_acc_matrix[:, 1:].flatten()  # Shape: (batch_size * (n_chunks - 1),)
        
        # # Align activities with labels: we only have labels for chunks 1 onwards
        # activity_for_classification = pooled_activity[:, 1:, :]  # Shape: (batch_size, n_chunks - 1, activity_dim)
        # activity_for_classification = activity_for_classification.reshape(-1, activity_for_classification.size(-1))  # Flatten to (batch_size * (n_chunks - 1), activity_dim)

        # Store the flattened data
        head_activities[address] = activity_for_classification.float().cpu().numpy()
        head_labels[address] = labels.float().cpu().numpy()


    # --- Classification ---
    results = {}
    for target in targets:
        if args.module == 'heads':
            layer, head = target
            address = f'{layer}-{head}'
        else:
            layer = target
            address = f'{layer}'

        X = head_activities[address] # (n_valid_chunks_total, activity_dim)
        y = head_labels[address]     # (n_valid_chunks_total,)

        if len(np.unique(y)) < 2:
            print(f"Skipping module {args.module} {address} due to single class in labels.")
            results[address] = 0.5
            continue

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
        
        results[address] = acc
        print(f"Module {args.module} {address}: Classification accuracy = {acc:.4f}, avg head acc: {y.mean().item()}, prediction_mean: {y_pred.mean()}")

    save_dir = Path('data/classification_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    model_str = args.model_name.split("/")[-1]
    torch.save(results, f'{save_dir}/{model_str}_{args.module}_classification_accs_finegrained.pt')
    print(f"Saved classification results to {save_dir}/{model_str}_{args.module}_classification_accs.pt")

if __name__ == '__main__':
    main()
