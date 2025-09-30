from pathlib import Path

import torch
from fastcore.script import call_parse
from matplotlib import pyplot as plt

def aggregate_accs(
    accs:torch.Tensor, # tensor of accuracies for different heads, shape `(n_heads, n_repetitions)`
    method:str='topk'  # the aggregation method to use: 'mean', 'max', or 'topk'
):
    "aggregate a tensor of accuracies across heads using `method`."
    if not isinstance(accs, torch.Tensor) or accs.nelement() == 0: return torch.tensor([])
    if method == 'mean': return accs.mean(dim=0)
    if method == 'max':
        max_idx = accs.mean(dim=-1).argmax()
        return accs[max_idx]
    if method == 'topk':
        k = min(5, len(accs))
        _, top_indices = accs.mean(dim=-1).topk(k)
        return accs[top_indices].mean(dim=0)
    raise ValueError(f"Unknown aggregation method: {method}")

def get_head_accs(
    path:Path,                     # directory path containing the score files for each head
    induction_scores:torch.Tensor, # `(n_layers, n_heads)` tensor of induction scores
    learning_scores:torch.Tensor,  # `(n_layers, n_heads)` tensor of learning scores
    filter_fn:callable             # function that takes (induction_score, learning_score) and returns a boolean
):
    "get all accuracies for heads that satisfy `filter_fn`."
    exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
    files = [f for f in path.ls() if f.name not in exceptions]
    
    selected_accs = []
    for f in files:
        head_address = f.name.split('_')[0]
        layer, head = map(int, head_address.split('-'))
        
        ind_score = induction_scores[layer, head]
        learn_score = learning_scores[layer, head]
        
        if filter_fn(ind_score, learn_score):
            accs = torch.load(f, weights_only=False).mean(dim=0)
            selected_accs.append(accs)

    return torch.stack(selected_accs) if selected_accs else torch.tensor([])

@call_parse
def main(
    model_name:str = 'Qwen/Qwen2.5-1.5B', # model name from Hugging Face
    agg:str = 'topk',                     # aggregation method: 'topk', 'mean', or 'max'
    thresh:float = 0.4                    # threshold for induction score
):
    "plot ICL learning scores"
    model_s = model_name.split("/")[-1]
    markov_ords = [2, 3]

    fig, ax = plt.subplots(2, 3, figsize=(6, 6), sharey=True, sharex=True)

    for i, markov_ord in enumerate(markov_ords):
        order = f'markov{markov_ord}'
        path = Path(f'data/learning_scores/{order}/{model_s}')
        if not path.exists(): continue

        exp_args = torch.load(path/'args.pt', weights_only=False)
        accs = torch.load(path/'model_accs.pt', weights_only=False)
        accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)
        accs = accs.view(accs.size(0), -1, exp_args.chunk_size).mean(dim=0)
        
        if markov_ord == 2: lm_accs = accs[:, :-1].mean(dim=1)
        else:
            mask = torch.arange(1, exp_args.chunk_size + 1) % (exp_args.chunk_size // exp_args.n_permute_primitive) == 0
            mask[-1] = False
            lm_accs = accs[:, mask].mean(dim=1)
        ax[i, 0].plot(lm_accs * 100, label=model_s)

        induction_scores = torch.load(f'data/induction_scores/{model_s}.pt', weights_only=False)
        learning_scores = torch.load(path/'learning_scores.pt', weights_only=False)

        induction_head_accs = get_head_accs(path, induction_scores, learning_scores, lambda ind_s, learn_s: ind_s > thresh)
        aggregated_induction_accs = aggregate_accs(induction_head_accs, agg)
        if aggregated_induction_accs.numel() > 0: ax[i, 1].plot(aggregated_induction_accs * 100, label=model_s)

        generic_head_accs = get_head_accs(path, induction_scores, learning_scores, lambda ind_s, learn_s: learn_s > 0.4 and ind_s < 0.4)
        aggregated_generic_accs = aggregate_accs(generic_head_accs, agg)
        if aggregated_generic_accs.numel() > 0: ax[i, 2].plot(aggregated_generic_accs * 100, label=model_s)
    
    ax[0, 0].set_title('LM Head')        
    ax[0, 1].set_title('Induction Heads')
    ax[0, 2].set_title('Generic\nICL Heads')
    ax[1, 1].set_xlabel('Repetitions')
    ax[0, 0].set_ylabel('Markov Order 2')
    ax[1, 0].set_ylabel('Markov Order 3')
    ax[0, 0].set_ylim([0., 100])
    
    fig.supylabel('Accuracy (%)')
    plt.tight_layout()
    plt.show()