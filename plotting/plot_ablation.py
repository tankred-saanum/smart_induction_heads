import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-0.5B', type=str)
    parser.add_argument('--ablation_style', default='induction', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=2, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()

fig, ax = plt.subplots(1, 3, figsize=(10, 6))


order='markov2' if args.markov_order==2 else 'markov3'
exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
for i, file in enumerate(files):
    if file in exceptions:
        continue

    accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
    accs = accs.mean(dim=0)

    ax[0].plot(accs, linewidth=1)
    ax[0].set_ylim([0., 1.1])

files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
for i, file in enumerate(files):
    if file in exceptions:
        continue

    accs = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
    accs = accs.mean(dim=0)

    ax[1].plot(accs, linewidth=1)
    ax[1].set_ylim([0., 1.1])

exp_args = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/args.pt', weights_only=False)
exp_args_ablated = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/args.pt', weights_only=False)
accs_ablated = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
accs_ablated = torch.cat([accs_ablated, torch.ones(accs.size(0), 1)], dim=-1)
accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)
accs_set = [accs, accs_ablated]
for accs in accs_set:
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
    ax[-1].plot(accs)
    ax[-1].set_ylim([0., 1.1])
plt.legend()
plt.show()






# layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
# for layer in layer_dict:

#     for head in layer_dict[layer]:
#         accs = torch.load(f'data/learning_scores/{args.model_name.split("/")[-1]}/{layer}-{head}_accs.pt', weights_only=False)
#         accs = accs.mean(dim=0)
#         ax[0].plot(accs, linewidth=1)
        
#         accs = torch.load(f'data/learning_scores_ablated/{args.model_name.split("/")[-1]}/{layer}-{head}_accs.pt', weights_only=False)
#         accs = accs.mean(dim=0)
#         ax[1].plot(accs, linewidth=1)
        
# accs_ablated = torch.load(f'data/learning_scores_ablated/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)[:, :-7]
# accs = torch.load(f'data/learning_scores/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)

# accs.shape
# ax[-1].plot(accs_ablated.mean(dim=0))
# ax[-1].plot(accs.mean(dim=0))
# plt.show()