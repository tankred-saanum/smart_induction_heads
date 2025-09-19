import torch
from argparse import ArgumentParser
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
args.module='heads'
# if True:
#     args = get_config()
#     args.module='heads'
#     order=2
#     exp_args = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
#     decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
#     decoding_accs[decoding_accs<0.9] = 0
#     plt.imshow(decoding_accs.T)
#     plt.show()

# f, ax = plt.subplots(1, 2, figsize=(10, 6))
# markov_orders = [2, 3]
# for i, order in enumerate(markov_orders):
#     exp_args = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
#     decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
#     ax[i].imshow(decoding_accs.T)

# cbar = f.colorbar(im1, ax=[ax1, ax2], shrink=0.8, aspect=20)
# cbar.set_label('Value')

# plt.show()

figsize = plt.rcParams['figure.figsize']
# Access individual values
standard_width = figsize[0]   # 6.99866
standard_height = figsize[1]  # 4.8
def plot_max_learning_scores():
    models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
    fig, ax = plt.subplots(1, len(models), figsize=(standard_width, standard_height/1.5), sharey=True)
    # for ax_i in ax:
    #     ax_i.set_box_aspect(2)  # or 'auto', or a number like 2.0
    
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
            scores = decoding_accs.max(dim=-1)[0]*100
            ax[i].plot(scores, color=colors[j], label=f'Order {order}')
            ax[i].set_ylim([45, 100.])
            
            ax[i].set_title(model_str)
    ax[0].set_ylabel('1-Back Accuracy')
    fig.supxlabel('Layer', y=-0.05)
    h, l = ax[-1].get_legend_handles_labels()
    fig.legend(h, l, ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    #fig.tight_layout()
    plt.savefig('figures/max_1back.png', bbox_inches='tight')
    fig.show()
plot_max_learning_scores()

# args.model_name
# order='3'
# args.module='heads'
# decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
# decoding_accs[15, 7]