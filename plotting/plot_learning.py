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

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
order='markov2' if args.markov_order==2 else 'markov3'
files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
exceptions = ['learning_scores.pt', 'model_accs.pt']
avgs = []
score=0
max_idx = 0
for i, file in enumerate(files):
    if file in exceptions:
        continue

    accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
    accs = accs.mean(dim=0)
    new_score = accs[-10:].mean()
    if new_score>score:
        score=new_score
        max_idx=i
    ax[0].plot(accs, linewidth=1)
    # avgs.append(accs)

#ax[0].plot(torch.stack(avgs)[max_idx], linewidth=1)
ax[0].set_ylim([0., 1.1])

accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)

ax[-1].plot(accs.mean(dim=0))
plt.show()