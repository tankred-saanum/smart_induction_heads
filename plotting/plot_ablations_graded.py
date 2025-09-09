import torch
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os
import matplotlib


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B', type=str)
    parser.add_argument('--ablation_style', default='one_back', type=str)
    parser.add_argument('--aggregate', default='topk', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)   
    parser.add_argument('--cutoff', default=0, type=int)
    parser.add_argument('--markov_order', default=2, type=int)   
    args, _ = parser.parse_known_args()

    return args


args = get_config()
markov_orders= [2, 3]

models = ['Qwen/Qwen2.5-1.5B']

fig, ax = plt.subplots(2, 3, figsize=(6, 6), sharey=True, sharex=True)



exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']

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
    ax[i, 0].plot(accs*100, label='No lesion')
    
    
    for threshold in [0.9, 0.85, 0.8]:
        args.threshold=threshold
        accs = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/model_accs.pt', weights_only=False)
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
        ax[i, 0].plot(accs*100, label=f'Accuracy > {args.threshold}')
        
        
        
    
    ax[i, 0].set_ylim([0., 100])
ax[0, 0].set_title('LM Head')
#ax[0, 0].set_ylabel('Accuracy %')
        
        


exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
for i, order in enumerate(markov_orders):
    order='markov2' if order==2 else 'markov3'
    
    model_str= args.model_name.split("/")[-1]
    induction_scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')
    files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
    aggregate = []
    for _, file in enumerate(files):
        if file in exceptions:
            continue
        head_address = file.split('_')[0]
        layer, head = head_address.split('-')
        induction_score = induction_scores[int(layer), int(head)]
        
        if induction_score > 0.4:
        
            accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
            accs = accs.mean(dim=0)
            aggregate.append(accs)
    accs = torch.stack(aggregate, dim=0)
    if args.aggregate == 'mean':
        accs = accs.mean(dim=0)
    elif args.aggregate == 'max':
        max_idx = accs.mean(dim=-1).argmax()
        accs = accs[max_idx]
    elif args.aggregate == 'topk':
        _, max_idx = accs.mean(dim=-1).topk(5)
        accs = accs[max_idx].mean(dim=0)
    ax[i, 1].plot(accs*100, label='No lesion')
    
    
    
    
    # now for the ablation
    for threshold in [0.9, 0.85, 0.8]:
        args.threshold=threshold
        files = os.listdir(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}')
        aggregate = []
        for _, file in enumerate(files):
            if file in exceptions:
                continue
            head_address = file.split('_')[0]
            layer, head = head_address.split('-')
            induction_score = induction_scores[int(layer), int(head)]
            
            if induction_score > 0.4:
            
                accs = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
                accs = accs.mean(dim=0)
                aggregate.append(accs)
        accs = torch.stack(aggregate, dim=0)
        if args.aggregate == 'mean':
            accs = accs.mean(dim=0)
        elif args.aggregate == 'max':
            max_idx = accs.mean(dim=-1).argmax()
            accs = accs[max_idx]
        elif args.aggregate == 'topk':
            _, max_idx = accs.mean(dim=-1).topk(5)
            accs = accs[max_idx].mean(dim=0)
        ax[i, 1].plot(accs*100, label=f'Accuracy > {args.threshold}')
        
    
    
#ax[1, 0].set_ylabel('Accuracy %')
ax[0, 1].set_title('Induction Heads')





exceptions = ['learning_scores.pt', 'model_accs.pt', 'args.pt']
for i, order in enumerate(markov_orders):
    order='markov2' if order==2 else 'markov3'
    model_str= args.model_name.split("/")[-1]
    learning_scores = torch.load(f'data/learning_scores/{order}/{model_str}/learning_scores.pt')
    induction_scores = torch.load(f'data/induction_scores/{args.model_name.split("/")[-1]}.pt')
    files = os.listdir(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}')
    aggregate = []
    for _, file in enumerate(files):
        if file in exceptions:
            continue
        head_address = file.split('_')[0]
        layer, head = head_address.split('-')
        learning_score = learning_scores[int(layer), int(head)]
        
        induction_score = induction_scores[int(layer), int(head)]
        if learning_score > 0.4 and induction_score < 0.4:
        
            accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
            accs = accs.mean(dim=0)
            aggregate.append(accs)
    accs = torch.stack(aggregate, dim=0)
    if args.aggregate == 'mean':
        accs = accs.mean(dim=0)
    elif args.aggregate == 'max':
        max_idx = accs.mean(dim=-1).argmax()
        accs = accs[max_idx]
    elif args.aggregate == 'topk':
        _, max_idx = accs.mean(dim=-1).topk(5)
        accs = accs[max_idx].mean(dim=0)
    ax[i, 2].plot(accs*100, label='No lesion')
    
    
    
    
    
    
    
    for threshold in [0.9, 0.85, 0.8]:
        args.threshold=threshold
        files = os.listdir(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}')
        aggregate = []
        for _, file in enumerate(files):
            if file in exceptions:
                continue
            head_address = file.split('_')[0]
            layer, head = head_address.split('-')
            learning_score = learning_scores[int(layer), int(head)]
            
            induction_score = induction_scores[int(layer), int(head)]
            if learning_score > 0.4 and induction_score < 0.4:
            
                accs = torch.load(f'data/ablated_learning_scores/{args.ablation_style}_threshold={args.threshold}/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
                accs = accs.mean(dim=0)
                aggregate.append(accs)
        accs = torch.stack(aggregate, dim=0)
        if args.aggregate == 'mean':
            accs = accs.mean(dim=0)
        elif args.aggregate == 'max':
            max_idx = accs.mean(dim=-1).argmax()
            accs = accs[max_idx]
        elif args.aggregate == 'topk':
            _, max_idx = accs.mean(dim=-1).topk(5)
            accs = accs[max_idx].mean(dim=0)
        ax[i, 2].plot(accs*100, label=f'Accuracy > {args.threshold}')
        
    
    
    
#ax[1, 0].set_ylabel('Accuracy %')
ax[0, 2].set_title('Context\nMatching Heads')



h, l = ax[1, -1].get_legend_handles_labels()
fig.legend(h, l, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 0.01))
fig.supxlabel('Repetitions')
fig.supylabel('Accuracy %')
plt.savefig(f'figures/ablation={args.ablation_style}_{model_str}_graded.png', bbox_inches='tight')
plt.show()

# avgs = []
# score=0
# max_idx = 0
# files = os.listdir(f'data/learning_scores/{order}/{model_name.split("/")[-1]}')
# for i, file in enumerate(files):
#     if file in exceptions:
#         continue

#     accs = torch.load(f'data/learning_scores/{order}/{args.model_name.split("/")[-1]}/{file}', weights_only=False)
#     accs = accs.mean(dim=0)

#     plt.plot(accs, linewidth=1)

# plt.show()
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