from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from transformers import PretrainedConfig


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-3B', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)   
    parser.add_argument('--markov_order', default=2, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()


# layer_dict = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}_{args.threshold}.pt')


config = PretrainedConfig.from_pretrained(args.model_name)
layer_dict = {}
for layer in range(config.num_hidden_layers):
    layer_dict[layer] = list(range(config.num_attention_heads))

scores = torch.zeros(config.num_hidden_layers, config.num_attention_heads)

for layer in layer_dict:

    for head in layer_dict[layer]:
        accs = torch.load(f'data/learning_scores/{args.model_name.split("/")[-1]}/{layer}-{head}_accs.pt', weights_only=False)
        accs = accs.mean(dim=0)
        final_acc = accs[-25:].mean()
        scores[layer, head]= final_acc

scores[scores<0.5]=0.0
plt.imshow(scores.T)
plt.colorbar()
plt.show()
scores.shape
accs = scores.topk(5)[0].mean(dim=1)
accs
accs.cummax(dim=0)
plt.plot(accs.cummax(dim=0)[0])
plt.show()