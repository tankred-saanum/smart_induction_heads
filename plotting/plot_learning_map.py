import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt

def get_config():
    parser = ArgumentParser()


    parser.add_argument('--markov_order', default=2, type=int)
    parser.add_argument('--threshold', default=0.4, type=float) 
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)   
    args, _ = parser.parse_known_args()

    return args
args = get_config()


fig, ax = plt.subplots(3, 1, sharex=True)
models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
colors = ['#8a2f08', '#2d7acc']
for i, model_name in enumerate(models):
    learning_scores2 = torch.load(f'data/learning_scores/markov2/{model_name.split("/")[-1]}/learning_scores.pt')
    learning_scores3 = torch.load(f'data/learning_scores/markov3/{model_name.split("/")[-1]}/learning_scores.pt')
    max_learning_scores2 = learning_scores2.max(dim=1)[0]

    max_learning_scores3 = learning_scores3.max(dim=1)[0]
    
    
    ax[i].set_title(f'{model_name.split("/")[-1].split("-")[-1]}', y=0.5, x=0.95)
    ax[i].plot(max_learning_scores2*100, label='2nd order', color=colors[0])
    ax[i].plot(max_learning_scores3*100, label='3nd order', color=colors[1])
fig.supylabel('Max attention\naccuracy %', x=-0.025)
fig.supxlabel('Layer')
h, l = ax[-1].get_legend_handles_labels()
fig.legend(h, l, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.01))
plt.savefig('figures/learning_heads_accuracy_per_layer.png', bbox_inches='tight')
plt.show()