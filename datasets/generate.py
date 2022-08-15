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


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="./DialogGPT-medium", help="Path, url or short name of the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu).")
    parser.add_argument("--data_path", type=str, default="./original", help="Path of the dataset.")
    parser.add_argument("--out_path", type=str, default="./generated", help="Path of responses generated.")

    parser.add_argument("--do_sample", action='store_true', help="Set to use sampling instead of greedy search.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum length of new generated tokens.")
    parser.add_argument("--min_new_tokens", type=int, default=1, help="Minimum length of new generated tokens.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--temperature", type=int, default=1, help="Sampling softmax temperature.")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering).")
    parser.add_argument("--top_p", type=float, default=0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering).")
    parser.add_argument("--num_turn", type=int, default=5, help="Number of dialog turns to be generated.")
    parser.add_argument("--num_pre", type=int, default=2, help="Number of preceding utterances used.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")
        return
    
    with open(args.data_path, 'r', encoding='UTF-8') as f:
        dataset = json.load(f)

    logger.info("Get pretrained model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    model.to(args.device)

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    all_generated_dialogs = []
    for d in tqdm(dataset, mininterval=1):
        origin_dialog = [dd['text'] for dd in d]
        generated_dialogs = []
        for i in range(len(origin_dialog)):
            local_dialog = origin_dialog[max(i-args.num_pre,0):i+1]
            history = tokenizer.eos_token.join(local_dialog)
            history = history + tokenizer.eos_token
            for step in range(args.num_turn):
                if step == 0:
                    chat_history_ids = tokenizer.encode(history, return_tensors='pt').to(args.device)
                now_len = chat_history_ids.shape[-1]
            
                chat_history_ids = model.generate(
                    chat_history_ids, 
                    max_new_tokens=args.max_new_tokens,
                    min_length=now_len+args.min_new_tokens,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=args.do_sample,
                    top_k=args.top_k, 
                    top_p=args.top_p,
                    temperature=args.temperature
                )

                out = tokenizer.decode(chat_history_ids[:, now_len:][0], skip_special_tokens=True)
                out = re.sub('\s+', ' ', out)
                local_dialog.append(out)
            generated_dialogs.append({"origin_utterance":origin_dialog[i], "generated_dialog":local_dialog})
        
        assert len(generated_dialogs) == len(origin_dialog)
        all_generated_dialogs.append(generated_dialogs)
    assert len(all_generated_dialogs) == len(dataset)

    with open(args.out_path, 'w', encoding="UTF-8") as f:
        json.dump(all_generated_dialogs, f, indent=4, ensure_ascii=False)
    logger.info("saving to {}".format(args.out_path))

if __name__ == "__main__":
    main()
