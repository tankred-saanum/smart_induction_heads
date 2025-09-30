import matplotlib.pyplot as plt
import json
import matplotlib as mpl
import torch
from fastcore.basics import AttrDict
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

_ = torch.set_grad_enabled(False)
def main(model_name:str='Qwen/Qwen2.5-1.5B',
         gl: int= 19, # adaptive head layer
         gh: int= 3, # adaptive head head
         bl: int= 2, # static head layer
         bh: int= 3 # static head head
         ):
    mpl.rcParams['mathtext.fontset'] = 'cm'


    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = json.load(open("data/language.json"))
    prompts = [AttrDict(p) for p in prompts]

    for idx in range(0, len(prompts),2):
        prompts[idx].prompt_id = idx
        prompts[idx+1].prompt_id = idx

    df = AttrDict(p=[],head=[],ty=[],prompt=[],prompt_id=[])
    for prompt in tqdm(prompts):
        tokenized_input = tokenizer.encode(prompt.prompt, return_tensors="pt").to(model.device)
        c_id, w_id = tokenizer.encode(prompt.c)[0], tokenizer.encode(prompt.w)[0]
        look_right, look_wrong = torch.where(tokenized_input[0] == c_id)[0], torch.where(tokenized_input[0] == w_id)[0]
        attns = model(tokenized_input, output_attentions=True).attentions
        df.p.append(attns[gl][0,gh,-1,look_right].mean().item())
        df.head.append("Adaptive")
        df.ty.append("Correct")
        df.p.append(attns[gl][0,gh,-1,look_wrong].mean().item())
        df.head.append("Adaptive")
        df.ty.append("Wrong")
        df.p.append(attns[bl][0,bh,-1,look_right].mean().item())
        df.head.append("Static") 
        df.ty.append("Correct")
        df.p.append(attns[bl][0,bh,-1,look_wrong].mean().item())
        df.head.append("Static") 
        df.ty.append("Wrong")
        df.prompt.extend([prompt.prompt] * 4)
        df.prompt_id.extend([prompt.prompt_id] * 4)

    df = pd.DataFrame(df)
    g_correct_p = df[(df["head"]=="Adaptive") & (df.ty=="Correct")].p
    g_incorrect_p = df[(df["head"]=="Adaptive") & (df.ty=="Wrong")].p
    b_correct_p = df[(df["head"]=="Static") & (df.ty=="Correct")].p
    b_incorrect_p = df[(df["head"]=="Static") & (df.ty=="Wrong")].p

    g_correct_attn, g_correct_ci = g_correct_p.mean(), 1.96 * g_correct_p.sem()
    g_incorrect_attn, g_incorrect_ci = g_incorrect_p.mean(), 1.96 * g_incorrect_p.sem()
    b_correct_attn, b_correct_ci = b_correct_p.mean(), 1.96 * b_correct_p.sem()
    b_incorrect_attn, b_incorrect_ci = b_incorrect_p.mean(), 1.96 * b_incorrect_p.sem()

    width = 0.35
    x = torch.arange(2)
    fig, ax = plt.subplots(1, 1)

    rects1 = ax.bar(x - width/2, [g_correct_attn, b_correct_attn], width, yerr=[g_correct_ci, b_correct_ci], capsize=5, label='Correct', color="#7F3517", edgecolor='black')
    rects2 = ax.bar(x + width/2, [g_incorrect_attn, b_incorrect_attn], width, yerr=[g_incorrect_ci, b_incorrect_ci], capsize=5, label='Wrong', color="#7F3517", edgecolor='black')

    # fontsize x-large
    ax.bar_label(rects1, labels=["Correct"] * len(rects1), padding=1.5, fontsize="x-large")
    ax.bar_label(rects2, labels=["Wrong"] * len(rects2), padding=1.5, fontsize="x-large")

    ax.set_ylabel('Attention')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([f'Adaptive ({gl}, {gh})', f'Static ({bl}, {bh})'])

    plt.tight_layout()
    plt.savefig("figures/language_example_many.pdf")
    plt.savefig("figures/language_example_many.png", bbox_inches='tight')
    plt.show()
