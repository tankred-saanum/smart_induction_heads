import torch
from collections import defaultdict

def create_LH_dict(heads_arr, threshold):
    # heads is L x H
    layer_dict = defaultdict(list)
    for layer in range(heads_arr.size(0)):
        for head in range(heads_arr.size(1)):
            if heads_arr[layer, head] > threshold:
                layer_dict[layer].append(head)
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

def unique_second_order_markov_sequence(tokens, args):
    perms = []
    used_perms_indices = set()
    while len(perms) < args.n_permute:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    ordered_sequence = torch.arange(args.n_reps*args.n_permute)%args.n_permute
    permuted_sequence = ordered_sequence[torch.randperm(args.n_reps*args.n_permute)]
    all_tokens = []
    for seq_id in permuted_sequence:
        all_tokens.append(perms[seq_id])

    all_tokens = torch.cat(all_tokens, dim=0)
    chunk_id=(torch.cdist(permuted_sequence.unsqueeze(-1).float(), permuted_sequence.unsqueeze(-1).float(), p=0) == 0).float().tril(diagonal=-1)
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


def unique_third_order_markov_sequence(tokens, args):
    
    # first order perm
    perms = []
    used_perms_indices = set()
    while len(perms) < args.n_permute_primitive:
        perm_idx = torch.randperm(args.chunk_size)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms_indices:
            used_perms_indices.add(perm_idx_tuple)
            perms.append(tokens[perm_idx])
        
    # second order perm
    perms2 = []
    used_perms2_indices = set()
    while len(perms2) < args.n_permute:
        perm_idx = torch.randperm(args.n_permute_primitive)
        perm_idx_tuple = tuple(perm_idx.tolist())
        if perm_idx_tuple not in used_perms2_indices:
            used_perms2_indices.add(perm_idx_tuple)
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
    