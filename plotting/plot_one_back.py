import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from transformers import PretrainedConfig
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




border_decoding_accs = torch.load(f'data/border_classification_results/{args.model_name.split("/")[-1]}_markov{2}_border_classification.pt', weights_only=False)['results']
border_decoding_accs.values()
def dict2array(results_dict, model_name):
    config = PretrainedConfig.from_pretrained(model_name)
    arr = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    
    for k, v in results_dict.items():
        l, h = k.split('-')
        l=int(l)
        h=int(h)
        arr[l, h] = v
    return arr
dict2array(border_decoding_accs, model_name=args.model_name)

def plot_max_learning_scoresv2():
    models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
    fig, ax = plt.subplots(2, len(models), figsize=(standard_width, standard_height/1.5), sharey=True)
    # for ax_i in ax:
    #     ax_i.set_box_aspect(2)  # or 'auto', or a number like 2.0
    
    markov_orders = [2, 3]
    colors = ['#8a2f08', '#2d7acc']

    for i, model_name in enumerate(models):
        model_str = model_name.split('/')[-1]
        for j, order in enumerate(markov_orders):
            label = f'3rd order' if order==3 else '2nd order'
            exp_args = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/args.pt', weights_only=False)
            decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
            scores = decoding_accs.max(dim=-1)[0]*100
            ax[0, i].plot(scores, color=colors[j], label=label)
            ax[0, i].set_ylim([45, 100.])
            
            ax[0, i].set_title(model_str)
            
            
            border_decoding_accs = torch.load(f'data/border_classification_results/{model_name.split("/")[-1]}_markov{order}_border_classification.pt', weights_only=False)['results']
            border_decoding_accs = dict2array(border_decoding_accs, model_name=model_name)
            max_border_scores = border_decoding_accs.max(dim=-1)[0]*100
            
            ax[1, i].plot(max_border_scores, color=colors[j], label=label)
            ax[1, i].set_ylim([45, 100.])
            
            if i == 0:
                ax[0, i].set_ylabel('1-Back accuracy')
                ax[1, i].set_ylabel('Border decoding\naccuracy')
    fig.supxlabel('Layer', y=-0.05)
    h, l = ax[1, -1].get_legend_handles_labels()
    fig.legend(h, l, ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    #fig.tight_layout()
    plt.savefig('figures/1back_border.png', bbox_inches='tight')
    fig.show()

plot_max_learning_scoresv2()
border_decoding_accs

# args.model_name
# order='3'
# args.module='heads'
# decoding_accs = torch.load(f'data/one_back_scores/markov{order}/{args.model_name.split("/")[-1]}/{args.module}/decoding_accuracies.pt', weights_only=False)
# decoding_accs[15, 7]