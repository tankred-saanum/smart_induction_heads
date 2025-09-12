import torch
from matplotlib import pyplot as plt

scores_mmlu = [65.6, 64.2, 66.6, 68.4]
scores_wino = [71.1,78.0, 77.4, 81.25]
scores_arc = [56.5,  60.0, 59.3, 62.5]

models = ['Qwen2.5-3B', 'Mistral 7B', 'Llama3-8B', 'Osmosis (ours)']
benches = ['MMLU', 'Winogrande', 'ARC-C']
scores = [scores_mmlu, scores_wino, scores_arc]
hatch = ['', '', '', 'o']

for i, (bench, score) in enumerate(zip(benches, scores)):
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.bar(models, score, color='#eb4034', hatch=hatch)
    ax.set_ylim([min(score) - 15, max(score)+10])
    ax.set_title(bench)
    plt.savefig(f'figures/{bench}.png', bbox_inches='tight')
    plt.show()

