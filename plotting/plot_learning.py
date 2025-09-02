import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=3, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
order='markov2' if args.markov_order==2 else 'markov3'
exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
for model_name in models:
    files = os.listdir(f'data/learning_scores/{order}/{model_name.split("/")[-1]}')
    exp_args = torch.load(f'data/learning_scores/{order}/{model_name.split("/")[-1]}/args.pt', weights_only=False)
    accs = torch.load(f'data/learning_scores/{order}/{model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
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
    ax.plot(accs, label=model_name)
ax.set_ylim([0., 1.1])
plt.legend()
plt.show()

# avgs = []
# score=0
# max_idx = 0
files = os.listdir(f'data/learning_scores/{order}/{model_name.split("/")[-1]}')
for i, file in enumerate(files):
    if file in exceptions:
        continue

    accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
    accs = accs.mean(dim=0)
    new_score = accs[-10:].mean()
    # if new_score>score:
    #     score=new_score
    #     max_idx=i
    plt.plot(accs, linewidth=1)
    #avgs.append(accs)
plt.show()
# ax[0].plot(torch.stack(avgs)[max_idx], linewidth=1)
# ax[0].set_ylim([0., 1.1])

# accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
# accs = torch.cat([accs, torch.ones(accs.size(0), 1)], dim=-1)

# if args.markov_order==2:
#     accs = accs.view(accs.size(0), -1, 8)
#     accs = accs.mean(dim=0)
#     accs = accs[:, :-1].mean(dim=1)
# elif args.markov_order ==3:
#     accs = accs.view(accs.size(0), -1, 4*4)
#     accs = accs.mean(dim=0)
#     mask = torch.arange(1, (4*4)+1) % 4 == 0
#     mask[-1] = False
#     accs = accs[:, mask]
#     accs = accs.mean(dim=1)
# ax[-1].plot(accs)
# ax[-1].set_ylim([0., 1.1])
# plt.show()