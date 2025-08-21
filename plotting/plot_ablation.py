import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()

fig, ax = plt.subplots(1, 3, figsize=(10, 6))

layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')
for layer in layer_dict:
    if layer <= args.cutoff:
        continue
    for head in layer_dict[layer]:
        accs = torch.load(f'data/learning_scores/{args.model_name.split("/")[-1]}/{layer}-{head}_accs.pt', weights_only=False)
        accs = accs.mean(dim=0)
        ax[0].plot(accs, linewidth=1)
        
        accs = torch.load(f'data/learning_scores_ablated/{args.model_name.split("/")[-1]}/{layer}-{head}_accs.pt', weights_only=False)
        accs = accs.mean(dim=0)
        ax[1].plot(accs, linewidth=1)
        
accs_ablated = torch.load(f'data/learning_scores_ablated/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)[:, :-7]
accs = torch.load(f'data/learning_scores/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)

accs.shape
ax[-1].plot(accs_ablated.mean(dim=0))
ax[-1].plot(accs.mean(dim=0))
plt.show()