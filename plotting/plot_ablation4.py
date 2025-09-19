import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os
import matplotlib


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--ablation_style', default='one_back', type=str)
    parser.add_argument('--random_ablation_name', default='random', type=str)
    parser.add_argument('--non_random_ablation', default='one_back', type=str)
    parser.add_argument('--aggregate', default='topk', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=2, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()
#head_type = 'One-back' if args.ablation_style=='one_back' else 'Random'
markov_orders= [2, 3]

models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
fig, ax = plt.subplots(2, len(models), sharey=True, sharex=True)



exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
colors = ['#8a2f08', '#2d7acc', '#eba134']
non_random_ablation_label = 'Induction heads lesioned' if args.non_random_ablation=='induction' else 'Context matching heads lesioned'
random_ablation_label = 'Random heads lesioned'
for j, model_name in enumerate(models):
    args.model_name = model_name
    for i, order in enumerate(markov_orders):
        order='markov2' if order==2 else 'markov3'
        #args.threshold=0.9 if order==2 else 0.7
        files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
        exp_args = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/args.pt', weights_only=False)
        accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
        accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)

        if args.markov_order==2:
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            accs = accs[:, :-1].mean(dim=1)
        elif args.markov_order ==3:
            print(exp_args.chunk_size)
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            mask = torch.arange(1, (exp_args.chunk_size)+1) % (exp_args.chunk_size//exp_args.n_permute_primitive) == 0
            mask[-1] = False
            accs = accs[:, mask]
            accs = accs.mean(dim=1)
        ax[i, j].plot(accs*100, label='No lesion', color=colors[0])
        
        
        
        accs = torch.load(f'data/ablated_learning_scores/{args.random_ablation_name}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
        accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)

        if args.markov_order==2:
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            accs = accs[:, :-1].mean(dim=1)
        elif args.markov_order ==3:
            print(exp_args.chunk_size)
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            mask = torch.arange(1, (exp_args.chunk_size)+1) % (exp_args.chunk_size//exp_args.n_permute_primitive) == 0
            mask[-1] = False
            accs = accs[:, mask]
            accs = accs.mean(dim=1)
        ax[i, j].plot(accs*100, label=f'{random_ablation_label}', color=colors[1])
        
        
            
        accs = torch.load(f'data/ablated_learning_scores/{args.non_random_ablation}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
        accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)

        if args.markov_order==2:
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            accs = accs[:, :-1].mean(dim=1)
        elif args.markov_order ==3:
            print(exp_args.chunk_size)
            accs = accs.view(accs.size(0), -1, exp_args.chunk_size)
            accs = accs.mean(dim=0)
            mask = torch.arange(1, (exp_args.chunk_size)+1) % (exp_args.chunk_size//exp_args.n_permute_primitive) == 0
            mask[-1] = False
            accs = accs[:, mask]
            accs = accs.mean(dim=1)
        ax[i, j].plot(accs*100, label=f'{non_random_ablation_label}', color=colors[2])
        
        
        
        
        ax[i, j].set_ylim([0., 100])
    ax[0, j].set_title(f'{args.model_name.split("/")[-1]}')


ax[0, 0].set_ylabel('2nd order')

ax[1, 0].set_ylabel('3nd order')
h, l = ax[-1, -1].get_legend_handles_labels()
fig.legend(h, l, ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0.01))
fig.supxlabel('Repetitions')
fig.supylabel('Accuracy %', x=-0.025)
plt.savefig(f'figures/ablation={args.non_random_ablation}.png', bbox_inches='tight')
plt.show()
