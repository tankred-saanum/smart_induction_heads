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
    