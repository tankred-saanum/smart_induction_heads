import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-3B', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)   
    args, _ = parser.parse_known_args()

    return args

args=get_config()
scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')
scores[scores<args.threshold] = 0.0
plt.imshow(scores.T)
plt.colorbar()
plt.xlabel('Layers')
plt.ylabel('Heads')
plt.title(f'Thresholded at {args.threshold}')
plt.show()