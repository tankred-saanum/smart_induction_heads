from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--module', default='heads', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=3, type=int)
    args, _ = parser.parse_known_args()
    return args
args= get_config()
figsize = plt.rcParams['figure.figsize']
standard_width = figsize[0]   # 6.99866
standard_height = figsize[1]  # 4.8
models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
fig, ax = plt.subplots(1, len(models), figsize=(standard_width, standard_height/2), sharey=True)
markov_orders = [2, 3]
colors = ['#8A2F08', '#2D7ACC']
lwd=3
spn_lwd=2.0
lbl_size = 18
for i, model_name in enumerate(models):
    model_str = model_name.split('/')[-1]
    for j, order in enumerate(markov_orders):
        exp_args = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
        decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
        scores = decoding_accs.max(dim=-1)[0]*100
        ax[i].plot(scores, color=colors[j], label=f'Order {order}')
        ax[i].set_ylim([45, 100.])
        ax[i].set_xticks(list(range(0, scores.size(0), 10)))
        ax[i].set_title(model_str, y=.9)
ax[0].set_ylabel('Context decodability', multialignment='center', y=0.4)
ax[1].set_xlabel('Layer')
h, l = ax[-1].get_legend_handles_labels()
fig.legend(h, l, ncols=2, loc='upper center', bbox_to_anchor=(0.55, .13), frameon=False)
fig.tight_layout()
plt.savefig('figures/1back.png', bbox_inches='tight')
plt.savefig('figures/1back.pdf', bbox_inches='tight')

