import json
from pathlib import Path

import einops
import torch
from fastcore.script import call_parse
from transformers import AutoModelForCausalLM

_ = torch.set_grad_enabled(False)
g = torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)

@call_parse
def main(
    model_name = "Qwen/Qwen2.5-1.5B", # hf mode id
    threshold = 0.4, # threshold for induction scores
):
    """
    Calculate the prefix matching score for a given model.
    The results are saved in a layers by head torch tensor, where each element corresponds to the induction score for a specific layer and head. The score is bound between 0 and 1, where 0 means no induction and 1 means perfect induction.

    Save the induction scores in `data/induction_scores` directory, with the model name as the filename. Additionally, save a JSON file with the induction scores above the specified threshold, where layers are keys and heads are values.
    """
    
    # save data here
    save_dir = Path("data/induction_scores")
    save_dir.mkdir(parents=True, exist_ok=True)
    

    # model stuff
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")


    # generate [<bos>, sequence, sequence]
    token_ids = einops.rearrange(torch.randint(low=1000,high=2000,size=(10,)), "seq_len -> 1 seq_len")
    token_seq = einops.repeat(token_ids, "batch seq_len -> batch (2 seq_len)").to(model.device).squeeze()
    token_seq = torch.concat([torch.Tensor([model.config.bos_token_id]).to(model.device).to(torch.long),token_seq])

    # run model
    out = model(token_seq.unsqueeze(0),return_dict=True,output_attentions=True)

    # create model attention matrices
    seq_len = len(token_seq)
    look_back = seq_len // 2 - 1
    induction_mask = torch.zeros(seq_len,seq_len).to(float)

    for example in range(seq_len//2+1, seq_len): # from +1 because of <bos> token
        induction_mask[example,example-look_back]=1.

    induction_mask = induction_mask[1:,1:] # ignore <bos>

    # compute induction scores
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    induction_scores = torch.zeros(num_layers,num_heads)
    tril = torch.tril_indices(seq_len-1,seq_len-1)
    induction_flat = induction_mask[tril[0],tril[1]].flatten()

    for layer in range(num_layers):
        for head in range(num_heads):
            pattern = out["attentions"][layer][0][head].cpu().to(float)[1:,1:] # ignore <bos>
            pattern_flat = pattern[tril[0], tril[1]].flatten()

            # dot product between the empirical and the model matrix, normalised by the total empirical attention
            induction_scores[layer, head] =  (induction_flat @ pattern_flat) / pattern_flat.sum()

        
    # save induction scores
    save_name = model_name.split("/")[-1]
    torch.save(induction_scores, save_dir / f"{save_name}.pt")

    # save the induction scores above the threshold where layers should be keys and heads should be values
    induction_heads = {}
    layers, heads = torch.where(induction_scores > threshold)
    for layer, head in zip(layers.tolist(), heads.tolist()):
        if layer not in induction_heads:
            induction_heads[layer] = []
        induction_heads[layer].append(head)

    with open(save_dir / f"{save_name}_{threshold}.json", "w") as f:
        json.dump(induction_heads, f, indent=4)