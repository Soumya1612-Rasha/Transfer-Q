from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification, AutoModel
import argparse
import torch
import json
import re
import os 
import pdb
import numpy as np
from tqdm import tqdm
np.random.seed(42)
torch.manual_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)


args = parser.parse_args()
model_name = "usvsnsp/pythia-6.9b-rm-full-hh-rlhf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

with open(args.out_file, "r") as out_f:
    lines = json.load(out_f)

rm_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.float16).to(args.rm_gpu)
rm_model = rm_model.to("cuda:0")
rm_model.eval()

def extract_out(output_data):
   
    if "response" in output_data:
        output = output_data["response"]
    elif "output" in output_data:
        output = output_data["output"]

    output_np = output.removeprefix(output_data["prompt"])
    if output_np.startswith(": "): output = output_np[2:]
    output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
    return output_data["prompt"]+output_np


def get_rm(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(args.rm_gpu)
    print(f"{tokens.shape=}")
    if tokens.shape[1] >= 1334: return None
    rm_out = rm_model(tokens)
    rm_val = rm_out.logits.flatten().item()
    del rm_out
    del tokens
    return rm_val

def get_rm_from_tokens(tokens):
    return rm_model(torch.tensor(tokens).unsqueeze(0).to(args.rm_gpu)).logits.flatten().item()



rm_scores = []
num_skip = 0
count = 0
for line in tqdm(lines):
    count += 1
    outp = extract_out(line)
    if len(outp) == 0: rm_scores.append(0.)
    rm_score = get_rm(outp)
    if rm_score == None: 
        print("skipped one")
        num_skip += 1
        continue
    else: rm_scores.append(rm_score)

import numpy as np
print(f"{np.mean(rm_scores)=}")
print(f"{num_skip=}")
