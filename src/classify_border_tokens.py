import warnings
from argparse import ArgumentParser
from pathlib import Path

import nnsight
import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    unique_second_order_markov_sequence,
    unique_third_order_markov_sequence,
)

warnings.filterwarnings("ignore")

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--total_batch_size', default=8, type=int)
    parser.add_argument('--n_permute', default=4, type=int)
    parser.add_argument('--n_permute_primitive', default=4, type=int)
    parser.add_argument('--chunk_size', default=8, type=int)
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--markov_order', default=2, type=int, choices=[2, 3])
    parser.add_argument('--n_reps', default=10, type=int, help='Number of repetitions of permutation patterns')
    args, _ = parser.parse_known_args()
    args.iters = args.total_batch_size // args.batch_size
    return args
def create_border_labels(tokens_shape, args):
    """
    Create binary labels for border tokens based on Markov order.
    
    For 2nd order: border tokens are at the end of highest-order chunks (size = chunk_size)
    For 3rd order: border tokens are at the end of highest-order chunks (size = chunk_size * n_permute_primitive)
    
    Args:
        tokens_shape: Shape of the token sequence (batch_size, seq_len)
        args: Configuration arguments
        
    Returns:
        torch.Tensor: Binary labels (1 for border tokens, 0 otherwise)
    """
    batch_size, seq_len = tokens_shape
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    if args.markov_order == 2:
        chunk_size = args.chunk_size
        for i in range(chunk_size - 1, seq_len, chunk_size):
            labels[:, i] = 1
            
    elif args.markov_order == 3:
        high_order_chunk_size = args.chunk_size * args.n_permute_primitive
        for i in range(high_order_chunk_size - 1, seq_len, high_order_chunk_size):
            labels[:, i] = 1
    
    return labels

def extract_head_outputs(model, all_batched_tokens, args):
    """
    Extract attention head outputs for all tokens in all sequences.
    
    Returns:
        dict: Dictionary mapping (layer, head) -> tensor of shape (total_tokens, head_dim)
    """
    config = model.config
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    
    head_outputs = {}
    
    # Process in batches
    for batch_idx in range(args.iters):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, all_batched_tokens.size(0))
        batch_tokens = all_batched_tokens[start_idx:end_idx]
        
        if batch_tokens.size(0) == 0:
            continue
            
        with torch.no_grad():
            with model.trace(batch_tokens, scan=False):
                saved_activations = {}
                for layer in range(config.num_hidden_layers):
                    saved_activations[layer] = model.model.layers[layer].self_attn.o_proj.input.save()
        
        for layer in range(config.num_hidden_layers):
            o_proj_input = saved_activations[layer].value  # batch, seq_len, hidden_size
            head_separated = rearrange(o_proj_input, 'b s (h d) -> b s h d', h=n_heads, d=head_dim)
            
            for head in range(n_heads):
                key = (layer, head)
                head_output = head_separated[:, :, head, :]  # batch, seq_len, head_dim
                
                if key not in head_outputs:
                    head_outputs[key] = []
                head_outputs[key].append(head_output.cpu())
    
    for key in head_outputs:
        head_outputs[key] = torch.cat(head_outputs[key], dim=0)  # total_batch_size, seq_len, head_dim
    
    return head_outputs

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nnsight.LanguageModel(args.model_name, device_map=device, torch_dtype=torch.bfloat16)
    config = model.config
    n_heads = config.num_attention_heads
    vocab_size = config.vocab_size


    all_batched_tokens = []
    all_border_labels = []
    
    for iter_idx in range(args.iters):
        batch_tokens = []
        
        for _ in range(args.batch_size):
            tokens = torch.randint(vocab_size, (args.chunk_size,))
            
            if args.markov_order == 2:
                all_tokens, _ = unique_second_order_markov_sequence(tokens, args)
            elif args.markov_order == 3:
                all_tokens, _ = unique_third_order_markov_sequence(tokens, args)
            else:
                raise ValueError(f"Unsupported Markov order: {args.markov_order}")
                
            batch_tokens.append(all_tokens)
        
        batch_tokens = torch.stack(batch_tokens)
        all_batched_tokens.append(batch_tokens)
        
        border_labels = create_border_labels(batch_tokens.shape, args)
        all_border_labels.append(border_labels)


    all_batched_tokens = torch.cat(all_batched_tokens, dim=0)
    all_border_labels = torch.cat(all_border_labels, dim=0)

    args.iters = (all_batched_tokens.size(0) + args.batch_size - 1) // args.batch_size
    head_outputs = extract_head_outputs(model, all_batched_tokens, args)
    
    flattened_labels = all_border_labels.flatten()
    
    # Leave-one-out cross-validation on sequence level
    n_sequences = all_batched_tokens.size(0)
    

    def get_repeat_indices(seq_len, args):
        """
        Get indices for each individual repeat based on sequence structure.
        
        Returns:
            dict: Dictionary mapping repeat_idx -> list of token indices for that repeat
        """
        if args.markov_order == 2:
            # For 2nd order: each chunk has size = chunk_size
            chunk_size = args.chunk_size
            total_chunks = seq_len // chunk_size
            n_chunks_per_repeat = args.n_permute
        elif args.markov_order == 3:
            # For 3rd order: each primitive chunk has size = chunk_size, 
            # but we group them into larger chunks of size = chunk_size * n_permute_primitive
            chunk_size = args.chunk_size * args.n_permute_primitive
            total_chunks = seq_len // chunk_size
            n_chunks_per_repeat = args.n_permute
        
        n_reps_actual = total_chunks // n_chunks_per_repeat
        
        repeat_indices = {}
        
        for repeat_idx in range(n_reps_actual):
            repeat_indices[repeat_idx] = []
            
            start_chunk = repeat_idx * n_chunks_per_repeat
            end_chunk = (repeat_idx + 1) * n_chunks_per_repeat
            
            for chunk_idx in range(start_chunk, min(end_chunk, total_chunks)):
                start_token = chunk_idx * chunk_size
                end_token = (chunk_idx + 1) * chunk_size
                repeat_indices[repeat_idx].extend(range(start_token, min(end_token, seq_len)))
        
        return repeat_indices

    results = {}
    repeat_results = {} 
    seq_len = all_batched_tokens.size(1)
    
    repeat_indices_dict = get_repeat_indices(seq_len, args)
    
    for layer in range(config.num_hidden_layers):
        for head in range(n_heads):
            key = (layer, head)
            head_address = f"{layer}-{head}"
            
            head_data = head_outputs[key].float().numpy()
            head_data_flat = head_data.reshape(-1, head_data.shape[-1])
            
            # Leave-one-out cross-validation
            overall_predictions = []
            overall_labels = []
            repeat_predictions = {repeat_idx: [] for repeat_idx in repeat_indices_dict.keys()}
            repeat_labels = {repeat_idx: [] for repeat_idx in repeat_indices_dict.keys()}
            
            for test_seq_idx in range(n_sequences):
                # Create train and test splits for this fold
                train_seq_indices = [i for i in range(n_sequences) if i != test_seq_idx]
                
                # Prepare training data
                train_tokens = []
                train_labels = []
                for seq_idx in train_seq_indices:
                    start_idx = seq_idx * seq_len
                    end_idx = (seq_idx + 1) * seq_len
                    train_tokens.append(head_data_flat[start_idx:end_idx])
                    train_labels.append(flattened_labels[start_idx:end_idx].numpy())
                
                X_train = np.concatenate(train_tokens, axis=0)
                y_train = np.concatenate(train_labels, axis=0)
                
                # Prepare test data for this fold
                test_start_idx = test_seq_idx * seq_len
                test_end_idx = (test_seq_idx + 1) * seq_len
                X_test = head_data_flat[test_start_idx:test_end_idx]
                y_test = flattened_labels[test_start_idx:test_end_idx].numpy()
                
                # Train the model on this fold
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(max_iter=1000, random_state=args.seed))
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                # Store overall predictions and labels
                overall_predictions.extend(y_pred)
                overall_labels.extend(y_test)
                
                # Store repeat-specific predictions and labels
                for repeat_idx, repeat_token_indices in repeat_indices_dict.items():
                    # Get test data for this repeat in this sequence
                    test_repeat_indices = [i for i in repeat_token_indices]
                    X_test_repeat = X_test[test_repeat_indices]
                    y_test_repeat = y_test[test_repeat_indices]
                    
                    y_pred_repeat = pipeline.predict(X_test_repeat)
                    
                    repeat_predictions[repeat_idx].extend(y_pred_repeat)
                    repeat_labels[repeat_idx].extend(y_test_repeat)
            
            # calculate overall accuracy from all cross-validation folds
            overall_acc = accuracy_score(overall_labels, overall_predictions)
            results[head_address] = overall_acc
            
            # calculate accuracy for each individual repeat
            repeat_results[head_address] = {}
            for repeat_idx in repeat_indices_dict.keys():
                if repeat_predictions[repeat_idx] and repeat_labels[repeat_idx]:
                    repeat_acc = accuracy_score(repeat_labels[repeat_idx], repeat_predictions[repeat_idx])
                    repeat_results[head_address][repeat_idx] = repeat_acc
                else:
                    repeat_results[head_address][repeat_idx] = None
            

    save_dir = Path('data/border_classification_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_str = args.model_name.split("/")[-1]
    filename = f'{model_str}_markov{args.markov_order}_border_classification.pt'
    save_path = save_dir / filename
    
    torch.save({
        'results': results,
        'repeat_results': repeat_results,  # Now contains per-repeat accuracies
        'args': vars(args),
        'model_name': args.model_name,
        'markov_order': args.markov_order,
        'border_fraction': all_border_labels.float().mean().item(),
        'n_reps_actual': len(repeat_indices_dict)  # Save actual number of repeats
    }, save_path)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 performing heads:")
    for i, (head_addr, acc) in enumerate(sorted_results[:10]):
        print(f"{i+1}. Head {head_addr}: {acc:.4f}")
    
    print(f"\nPer-repeat accuracies for top 5 heads (Total repeats: {len(repeat_indices_dict)}):")
    print("Head     | Overall | " + " | ".join([f"Rep{i:2d}" for i in range(len(repeat_indices_dict))]))
    print("-" * (20 + 8 * len(repeat_indices_dict)))
    
    for i, (head_addr, acc) in enumerate(sorted_results[:5]):
        repeat_accs = repeat_results[head_addr]
        
        repeat_strs = []
        for repeat_idx in range(len(repeat_indices_dict)):
            repeat_acc = repeat_accs.get(repeat_idx, None)
            if repeat_acc is not None:
                repeat_strs.append(f"{repeat_acc:.3f}")
            else:
                repeat_strs.append(" N/A ")
        
        repeat_line = " | ".join(repeat_strs)
        print(f"{head_addr:>8} | {acc:>7.4f} | {repeat_line}")
    

if __name__ == '__main__':
    main()
