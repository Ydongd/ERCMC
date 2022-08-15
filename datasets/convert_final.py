from math import trunc
import os
import logging
import random
from itertools import chain
from argparse import ArgumentParser
from pprint import pformat
import json
import re
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

def main(name):
    parser = ArgumentParser()
    parser.add_argument("--original_path", type=str, default="./original", help="Path of the original dataset.")
    parser.add_argument("--generated_path", type=str, default="./generated", help="Path of the generated dataset.")
    parser.add_argument("--final_path", type=str, default="./final", help="Path of the final dataset.")
    parser.add_argument("--num_pre", type=int, default=2, help="Number of preceding utterances used.")
    args = parser.parse_args()

    original_path_ = args.original_path
    generated_path_ = args.generated_path
    final_path_ = args.final_path
    original_path_ = os.path.join(original_path_, name)
    generated_path_ = os.path.join(generated_path_, name)
    final_path_ = os.path.join(final_path_, name)

    splits = ["train.json", "dev.json", "test.json"]

    for split in splits:
        original_path = os.path.join(original_path_, split)
        split_ = "out_" + split
        generated_path = os.path.join(generated_path_, split_)
        final_path = os.path.join(final_path_, split)
        with open(original_path, 'r', encoding='UTF-8') as f:
            origin_data = json.load(f)
        with open(generated_path, 'r', encoding='UTF-8') as f:
            generated_data = json.load(f)
        assert len(origin_data) == len(generated_data)
        final_data = []

        for i in range(len(origin_data)):
            origin = origin_data[i]
            generated = generated_data[i]
            assert len(origin) == len(generated)
            d_final = []
            for j in range(len(origin)):
                pos = j - max(j-args.num_pre, 0)
                assert origin[j]['text'] == generated[j]['origin_utterance'] == generated[j]['generated_dialog'][pos]
                assert len(generated[j]['generated_dialog'][pos+1:]) == 5
                d_final.append({"text":origin[j]['text'], 
                                "speaker":origin[j]['speaker'], 
                                "label":origin[j]['label'],
                                "generated":generated[j]['generated_dialog'][pos+1:]})
            final_data.append(d_final)
        
        with open(final_path, 'w', encoding="UTF-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print("saving to {}".format(final_path))

if __name__ == "__main__":
    main("dailydialog")
