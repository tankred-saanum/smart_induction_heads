from pathlib import Path

import torch
from fastcore.script import call_parse
from matplotlib import pyplot as plt


@call_parse
def main(
    model_name:str='Qwen/Qwen2.5-1.5B', # model name from Hugging Face
    module:str='heads'                 # module type, e.g., 'heads'
):
    "generate and save a plot of context decodability for a single model."
    model_s = model_name.split('/')[-1]
    markov_ords = [2, 3]
    colors = ['#8A2F08', '#2D7ACC']
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    max_layers = 0
    for i, order in enumerate(markov_ords):
        path = Path(f'data/one_back_scores/markov{order}/{model_s}/{module}/decoding_accuracies.pt')

        decoding_accs = torch.load(path, weights_only=False)
        scores = decoding_accs.max(dim=-1)[0] * 100
        max_layers = max(max_layers, scores.size(0))
        
        ax.plot(scores, color=colors[i], label=f'Order {order}', linewidth=2.5)
        
    ax.set_ylim([45, 100.])
    ax.set_xticks(list(range(0, max_layers, 5)))
    
    ax.set_ylabel('Context Decodability (%)')
    ax.set_xlabel('Layer')
    ax.legend(ncols=2, loc='lower center', bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    fig.tight_layout()
    plt.show()