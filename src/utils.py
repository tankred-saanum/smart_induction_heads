import torch

def first_order_markov_sequence(tokens, args):
    seq = tokens[torch.randperm(args.chunk_size)]
    seq = seq.repeat(args.n_reps*args.n_permute)
    return seq


def second_order_markov_sequence(tokens, args):
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
    chunk_id=(torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    return all_tokens, chunk_id

def unique_second_order_markov_sequence(tokens, args, return_perms=False):
    """
    Generates a sequence of tokens based on a second-order Markov structure.

    Returns:
        all_tokens (Tensor): The concatenated sequence of all tokens.
        chunk_id (Tensor): A matrix indicating which tokens belong to the same original chunk.
        permuted_sequence (Tensor): The sequence of indices of the permutations used.
        chunked_sequence (Tensor): The sequence of permutations (chunks) as they appear.
    """
    perms = []
    used_perms_indices = set()
    # Generate unique permutations of the input tokens
    while len(perms) < args.n_permute:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # Create a random sequence of these unique permutations
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    chunked_sequence_list = []
    for seq_id in permuted_sequence:
        chunked_sequence_list.append(perms[seq_id])

    # Stack the list of chunks into a single tensor
    chunked_sequence = torch.stack(chunked_sequence_list, dim=0)
    
    # Flatten the sequence for other uses
    all_tokens = torch.cat(chunked_sequence_list, dim=0)
    
    # Calculate chunk_id for identifying tokens from the same original permutation instance
    chunk_id = (torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        return all_tokens, chunk_id, chunked_sequence
    return all_tokens, chunk_id
    

def third_order_markov_sequence(tokens, args):
    perms = []
    for _ in range(args.n_permute_primitive):
        perm_idx = torch.randperm(args.chunk_size)
        perms.append(tokens[perm_idx])
        
    # second order perm
    perms2 = []
    for _ in range(args.n_permute):
        perm_idx = torch.randperm(args.n_permute_primitive)
        _perm = []
        for idx in perm_idx:
            _perm.append(perms[idx])
        _perm = torch.cat(_perm, dim=0)
        perms2.append(_perm)

    ordered_sequence = torch.arange(args.n_reps*args.n_permute)%args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps*args.n_permute)]
    all_tokens = []
    for seq_id in permuted_sequence:
        all_tokens.append(perms2[seq_id])

    all_tokens = torch.cat(all_tokens, dim=0)
    chunk_id=(torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    return all_tokens, chunk_id


def unique_third_order_markov_sequence(tokens, args, return_perms=False):
    """
    Generates a sequence of tokens based on a third-order Markov structure.
    Optionally returns detailed permutation and chunk information for both
    high-order and primitive levels.
    """
    
    # --- First-order permutations (primitives) ---
    perms = []
    used_perms_indices = set()
    # Note: args.chunk_size here is assumed to be the size of the primitive chunk.
    while len(perms) < args.n_permute_primitive:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # --- Second-order permutations (sequences of primitives) ---
    perms2 = []
    primitive_compositions = [] 
    used_perms2_indices = set()
    while len(perms2) < args.n_permute:
        perm_idx = torch.randperm(args.n_permute_primitive)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms2_indices:
            used_perms2_indices.add(perm_idx_tuple)
            primitive_compositions.append(perm_idx)
            _perm = torch.cat([perms[idx] for idx in perm_idx], dim=0)
            perms2.append(_perm)

    # --- Create the final sequence by permuting the second-order chunks ---
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    high_order_permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    high_order_chunked_list = []
    primitive_permuted_list = []
    for seq_id in high_order_permuted_sequence:
        high_order_chunked_list.append(perms2[seq_id])
        primitive_permuted_list.append(primitive_compositions[seq_id])

    # Flatten the sequence for the primary return value
    all_tokens = torch.cat(high_order_chunked_list, dim=0)
    
    # Calculate chunk_id for identifying tokens from the same high-order chunk
    chunk_id = (torch.cdist(high_order_permuted_sequence.unsqueeze(-1).float(), high_order_permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        # Stack high-order chunks into a single tensor
        high_order_chunked_sequence = torch.stack(high_order_chunked_list, dim=0)
        
        # Concatenate to get the sequence of primitive chunk indices
        primitive_permuted_sequence = torch.cat(primitive_permuted_list, dim=0)

        # Create the chunked sequence at the primitive level
        primitive_chunked_list = [perms[i] for i in primitive_permuted_sequence]
        primitive_chunked_sequence = torch.stack(primitive_chunked_list, dim=0)
        
        return all_tokens, chunk_id, high_order_chunked_sequence, primitive_chunked_sequence
        
    return all_tokens, chunk_id
    

def get_chunk_ids_in_order(chunked_sequence):
    """
    Takes a chunked sequence tensor and returns a list of unique IDs
    representing the chunks in the order they appear.

    Args:
        chunked_sequence (torch.Tensor): A 2D tensor where each row is a chunk.

    Returns:
        list: A list of integer IDs for each chunk in the sequence.
    """
    unique_chunks_map = {}
    chunk_ids = []
    next_id = 0
    
    # Convert each row (chunk) to a hashable tuple to find unique chunks
    for chunk in chunked_sequence:
        chunk_tuple = tuple(chunk.tolist())
        
        # If the chunk hasn't been seen before, assign it a new ID
        if chunk_tuple not in unique_chunks_map:
            unique_chunks_map[chunk_tuple] = next_id
            next_id += 1
        
        # Append the ID of the current chunk to the results
        chunk_ids.append(unique_chunks_map[chunk_tuple])
        
    return chunk_ids

def get_chunks(A, args):
    n_total_chunks = args.n_permute * args.n_reps
    B = torch.zeros(n_total_chunks, n_total_chunks)
    for i in range(n_total_chunks):
        for j in range(n_total_chunks):
            chunk = A[(i*args.chunk_size):(i+1)*args.chunk_size, (j*args.chunk_size):(j+1)*args.chunk_size]
            B[i, j] = chunk.mean()
    return B


def get_chunks_3rd_order(A, args):
    """
    Processes a single attention map (A) to extract transition scores for both
    higher-order and primitive chunks.

    Args:
        A (torch.Tensor): The attention map tensor of shape (seq_len, seq_len).
        args: An object containing necessary parameters.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - B_high_order: A 2D tensor for high-order chunk transitions.
            - B_primitive: A 2D tensor for primitive chunk transitions.
    """
    # --- 1. Calculate High-Order Chunk Transitions (existing logic) ---
    n_high_order_chunks = args.n_permute * args.n_reps
    B_high_order = torch.zeros(n_high_order_chunks, n_high_order_chunks)
    primitive_chunk_size = args.chunk_size // args.n_permute_primitive
    
    for i in range(n_high_order_chunks):
        for j in range(n_high_order_chunks):
            # Select rows corresponding to the i-th high-order chunk
            rows = A[(i * args.chunk_size):((i + 1) * args.chunk_size), :]
            
            # Create a mask to select transition points (ends of primitive chunks)
            transition_idx = torch.arange(1, args.chunk_size + 1)
            mask = transition_idx % primitive_chunk_size == 0
            mask[-1] = False  # Exclude the last position of the high-order chunk
            
            # Apply the mask to get only the transition rows
            rows_at_transitions = rows[mask]
            
            # Select columns corresponding to the j-th high-order chunk
            patch_score = rows_at_transitions[:, (j * args.chunk_size):((j + 1) * args.chunk_size)]
            
            # Calculate the mean score for this transition
            B_high_order[i, j] = patch_score.mean()

    # --- 2. Calculate Primitive Chunk Transitions ---
    n_primitive_chunks = n_high_order_chunks * args.n_permute_primitive
    B_primitive = torch.zeros(n_primitive_chunks, n_primitive_chunks)

    for i in range(n_primitive_chunks):
        # The transition row is the last token of the i-th primitive chunk
        transition_row_idx = (i + 1) * primitive_chunk_size - 1
        
        # Ensure we don't go past the sequence length for the last row
        if transition_row_idx >= A.shape[0] -1:
            continue

        row = A[transition_row_idx, :]

        for j in range(n_primitive_chunks):
            # Select columns corresponding to the j-th primitive chunk
            start_col = j * primitive_chunk_size
            end_col = (j + 1) * primitive_chunk_size
            patch_score = row[start_col:end_col]
            
            # Calculate the mean score and store it
            B_primitive[i, j] = patch_score.mean()
            
    return B_high_order, B_primitive