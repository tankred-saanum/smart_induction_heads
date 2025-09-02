import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt
def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)
    parser.add_argument('--module', default='heads', type=str)
    
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=3, type=int)   
    args, _ = parser.parse_known_args()

    return args

args = get_config()
f, ax = plt.subplots(1, 2, figsize=(10, 6))
markov_orders = [2, 3]
for i, order in enumerate(markov_orders):
    exp_args = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
    decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
    ax[i].imshow(decoding_accs.T)

cbar = f.colorbar(im1, ax=[ax1, ax2], shrink=0.8, aspect=20)
cbar.set_label('Value')

plt.show()
def plot_max_learning_scores():
    models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
    fig, ax = plt.subplots(1, len(models), figsize=(12, 6), sharey=True)
    markov_orders = [2, 3]
    colors = ['#8a2f08', '#2d7acc']
    lwd=3
    spn_lwd=2.0
    lbl_size = 18
    for i, model_name in enumerate(models):
        model_str = model_name.split('/')[-1]
        for j, order in enumerate(markov_orders):
            exp_args = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
            decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
            scores = decoding_accs.max(dim=-1)[0]
            ax[i].plot(scores, color=colors[j], linewidth=lwd, label=f'Order {order}')
            ax[i].set_ylim([0.4, 1.])
            
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_linewidth(spn_lwd)
            ax[i].spines['bottom'].set_linewidth(spn_lwd)
            ax[i].tick_params(labelsize=14, size=8, width=1)
            ax[i].set_title(model_str, fontsize=lbl_size)
    ax[0].set_ylabel('1-Back Accuracy', size=lbl_size)
    fig.supxlabel('Layer', size=lbl_size)
    h, l = ax[-1].get_legend_handles_labels()
    fig.legend(h, l, ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0.01), fontsize=lbl_size)
    #fig.tight_layout()
    plt.savefig('figures/max_1back.png', bbox_inches='tight')
    fig.show()
plot_max_learning_scores()
