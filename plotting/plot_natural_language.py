from fastcore.script import call_parse
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ = torch.set_grad_enabled(False)

@call_parse
def main(model_name:str='Qwen/Qwen2.5-1.5B',
         gl: int= 19, # adaptive head layer
         gh: int= 3, # adaptive head head
         bl: int= 2, # static head layer
         bh: int= 3 # static head head
         ):

    mpl.rcParams['mathtext.fontset'] = 'cm'

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "I visited San Antonio and saw the Alamo, and San Francisco where I saw the Golden Gate bridge. After seeing the Alamo I realized how much I liked San"
    toi = (" Antonio", " Francisco")
    gl, gh = 19, 3
    bl, bh = 2, 3
    fig, ax = plt.subplots(1, 1)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attns = model(**inputs, output_attentions=True).attentions

    correct_ids = tokenizer(toi[0], add_special_tokens=False).input_ids
    incorrect_ids = tokenizer(toi[1], add_special_tokens=False).input_ids

    # find the locations of the correct and incorrect tokens in the input
    correct_locs = [j for j, token_id in enumerate(inputs.input_ids[0]) if token_id in correct_ids]
    incorrect_locs = [j for j, token_id in enumerate(inputs.input_ids[0]) if token_id in incorrect_ids]

    # how much attention does the head pay to the toi tokens from the last token
    g_correct_attn = attns[gl][0, gh, -1].cpu().float()[correct_locs].sum().item()
    g_incorrect_attn = attns[gl][0, gh, -1].cpu().float()[incorrect_locs].sum().item()

    b_correct_attn = attns[bl][0, bh, -1].cpu().float()[correct_locs].sum().item()
    b_incorrect_attn = attns[bl][0, bh, -1].cpu().float()[incorrect_locs].sum().item()


    width = 0.35
    x = torch.arange(2)

    rects1 = ax.bar(x - width/2, [g_correct_attn, b_correct_attn], width, label='Antonio', color="#7F3517", edgecolor='black')
    rects2 = ax.bar(x + width/2, [g_incorrect_attn, b_incorrect_attn], width, label='Francisco', color="#7F3517", edgecolor='black')

    # fontsize x-large
    ax.bar_label(rects1, labels=[toi[0].strip()] * len(rects1), padding=1.5, fontsize="x-large")
    ax.bar_label(rects2, labels=[toi[1].strip()] * len(rects2), padding=1.5, fontsize="x-large")

    ax.set_ylabel('Attention')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xticklabels([f'Adaptive ({gl}, {gh})', f'Static ({bl}, {bh})'])
    base_text = "'I visited San Antonio and saw the Alamo, and San\nFrancisco where I saw the Golden Gate Bridge. After\nseeing the Alamo I realized how much I liked "
    bold_word = r"$\boldsymbol{San}$"
    end_text = "'"

    full_text = base_text + bold_word + end_text

    ax.text(0.05, .8, full_text, fontsize="x-large", ha='left', va='bottom', wrap=True, transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='#7F3517', alpha=0.3))
    plt.tight_layout()
    plt.savefig("figures/language_example.pdf")
    plt.savefig("figures/language_example.png", bbox_inches='tight')
    plt.savefig("figures/language_example.svg", bbox_inches='tight')