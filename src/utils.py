from collections import defaultdict

import torch


def create_LH_dict(heads_arr, threshold):
    # heads is L x H
    layer_dict = defaultdict(list)
    for layer in range(heads_arr.size(0)):
        for head in range(heads_arr.size(1)):
            if heads_arr[layer, head] > threshold:
                layer_dict[layer].append(head)
    return layer_dict


def get_best_and_worst(heads_arr, induction_scores, threshold):
    # heads is L x H
    best_score = 0.0
    worst_score = 1.0
    layer_dict = defaultdict(list)
    for layer in range(heads_arr.size(0)):
        for head in range(heads_arr.size(1)):
            score = heads_arr[layer, head]
            induction_score = induction_scores[layer, head]
            if score > threshold and induction_score > threshold:
                if score > best_score:
                    best_score = score
                    best_address = (layer, head)
                if score < worst_score:
                    worst_score = score
                    worst_address = (layer, head)
                
    return best_address, worst_address

def create_random_dict(heads_arr, threshold, pool_threshold):
    # heads is L x H
    above = heads_arr > threshold
    below = heads_arr < threshold

    layer_dict = defaultdict(list)
    rand_idx = torch.randperm(heads_arr.size(0)* heads_arr.size(1))
    coords = torch.unravel_index(rand_idx, heads_arr.shape)
    num_heads = 0
    target_heads = above.float().sum().int()
    
    for idx in rand_idx:
        coord_x, coord_y = coords[0][idx].item(), coords[1][idx].item()
        
        if heads_arr[coord_x, coord_y] > pool_threshold:
            continue
        else:
            layer_dict[coord_x].append(coord_y)
            num_heads += 1
        if num_heads == target_heads:
            break
    
    return layer_dict


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
    """
    perms = []
    used_perms_indices = set()
    # make permutations unique
    while len(perms) < args.n_permute:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # create a random sequence of these unique permutations
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    chunked_sequence_list = []
    for seq_id in permuted_sequence:
        chunked_sequence_list.append(perms[seq_id])

    
    chunked_sequence = torch.stack(chunked_sequence_list, dim=0)
       
    all_tokens = torch.cat(chunked_sequence_list, dim=0)
    
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
    Generates a sequence of tokens with 3rd-order structure.
    """
    
    # second order chunks
    perms = []
    used_perms_indices = set()
    # Note: args.chunk_size here is assumed to be the size of the primitive chunk.
    while len(perms) < args.n_permute_primitive:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # 3rd order chunks
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

    # shuffle their order
    ordered_sequence = torch.arange(args.n_reps * args.n_permute) % args.n_permute
    high_order_permuted_sequence = ordered_sequence[torch.randperm(args.n_reps * args.n_permute)]
    
    high_order_chunked_list = []
    primitive_permuted_list = []
    for seq_id in high_order_permuted_sequence:
        high_order_chunked_list.append(perms2[seq_id])
        primitive_permuted_list.append(primitive_compositions[seq_id])

    all_tokens = torch.cat(high_order_chunked_list, dim=0)
    
    chunk_id = (torch.cdist(high_order_permuted_sequence.unsqueeze(-1).float(), high_order_permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
    
    if return_perms:
        # for the overview figure
        high_order_chunked_sequence = torch.stack(high_order_chunked_list, dim=0)
        
        primitive_permuted_sequence = torch.cat(primitive_permuted_list, dim=0)

        primitive_chunked_list = [perms[i] for i in primitive_permuted_sequence]
        primitive_chunked_sequence = torch.stack(primitive_chunked_list, dim=0)
        
        return all_tokens, chunk_id, high_order_chunked_sequence, primitive_chunked_sequence
        
    return all_tokens, chunk_id
    


def get_chunk_ids_in_order(chunked_sequence):
    """
    Takes a chunked sequence tensor and returns a list of unique IDs
    representing the chunks in the order they appear.
    """
    unique_chunks_map = {}
    chunk_ids = []
    next_id = 0
    
    # convert each row (chunk) to a hashable tuple to find unique chunks
    for chunk in chunked_sequence:
        chunk_tuple = tuple(chunk.tolist())
        
        
        if chunk_tuple not in unique_chunks_map:
            unique_chunks_map[chunk_tuple] = next_id
            next_id += 1
        
        
        chunk_ids.append(unique_chunks_map[chunk_tuple])
        
    return chunk_ids
