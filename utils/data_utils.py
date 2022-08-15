import logging
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, original_utterances, generated_utterances, speakers, labels):
        self.guid = guid
        self.original_utterances = original_utterances
        self.generated_utterances = generated_utterances
        self.speakers = speakers
        self.labels = labels

# support roberta only (not including token_type_ids)
class ERCDataset(Dataset):
    def __init__(self, all_original_input_ids, all_original_attention_mask, all_generated_input_ids, all_generated_attention_mask, all_speakers, all_label_ids):
        self.all_original_input_ids = all_original_input_ids
        self.all_original_attention_mask = all_original_attention_mask
        self.all_generated_input_ids = all_generated_input_ids
        self.all_generated_attention_mask = all_generated_attention_mask
        self.all_speakers = all_speakers
        self.all_label_ids = all_label_ids

    def __getitem__(self, index):
        dialog_input_ids = self.all_original_input_ids[index]
        dialog_attention_mask = self.all_original_attention_mask[index]
        dialog_generated_input_ids = self.all_generated_input_ids[index]
        dialog_generated_attention_mask = self.all_generated_attention_mask[index]
        dialog_speakers = self.all_speakers[index]
        dialog_label_ids = self.all_label_ids[index]
        return dialog_input_ids, dialog_attention_mask, dialog_generated_input_ids, dialog_generated_attention_mask, dialog_speakers, dialog_label_ids

    def __len__(self):
        # return the number of dialogues
        return len(self.all_label_ids)
    
def collate_fn(batch):
    batch_dialog_input_ids = []
    batch_dialog_attention_mask = []
    batch_dialog_generated_input_ids = []
    batch_dialog_generated_attention_mask = []
    batch_dialog_speakers = []
    batch_dialog_label_ids = []
    for i in range(len(batch)):
        batch_dialog_input_ids.append(batch[i][0])
        batch_dialog_attention_mask.append(batch[i][1])
        batch_dialog_generated_input_ids.append(batch[i][2])
        batch_dialog_generated_attention_mask.append(batch[i][3])
        batch_dialog_speakers.append(batch[i][4])
        batch_dialog_label_ids.append(batch[i][5])
    
    # cut off orginal utterances
    for i in range(len(batch_dialog_attention_mask)):
        max_len = 0
        for j in range(len(batch_dialog_attention_mask[i])):
            max_len = max(max_len, sum(batch_dialog_attention_mask[i][j]))
        for j in range(len(batch_dialog_attention_mask[i])):
            batch_dialog_input_ids[i][j] = batch_dialog_input_ids[i][j][:max_len]
            batch_dialog_attention_mask[i][j] = batch_dialog_attention_mask[i][j][:max_len]
    
    # cut off generated utterances
    for i in range(len(batch_dialog_generated_attention_mask)):
        for j in range(len(batch_dialog_generated_attention_mask[i])):
            max_len = 0
            for k in range(len(batch_dialog_generated_attention_mask[i][j])):
                max_len = max(max_len, sum(batch_dialog_generated_attention_mask[i][j][k]))
            for k in range(len(batch_dialog_generated_attention_mask[i][j])):
                batch_dialog_generated_input_ids[i][j][k] = batch_dialog_generated_input_ids[i][j][k][:max_len]
                batch_dialog_generated_attention_mask[i][j][k] = batch_dialog_generated_attention_mask[i][j][k][:max_len]

    return batch_dialog_input_ids, batch_dialog_attention_mask, batch_dialog_generated_input_ids, batch_dialog_generated_attention_mask, batch_dialog_speakers, batch_dialog_label_ids


def process_sentence(
    sentence,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    tokens = tokenizer.tokenize(sentence)
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
    tokens += [sep_token]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
    # add cls token
    if cls_token_at_end:
        tokens += [cls_token]
    else:
        tokens = [cls_token] + tokens
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return input_ids, input_mask


def convert_examples_to_features(
    examples,
    data_processor,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    all_original_input_ids = []
    all_original_attention_mask = []
    all_generated_input_ids = []
    all_generated_attention_mask = []
    all_speakers = []
    all_label_ids = []

    for example in examples:
        guid = example.guid
        original_utterances = example.original_utterances
        generated_utterances = example.generated_utterances
        speakers = example.speakers
        labels = example.labels

        dialog_input_ids = []
        dialog_attention_mask = []
        dialog_generated_input_ids = []
        dialog_generated_attention_mask = []
        dialog_speakers = []
        dialog_label_ids = []

        assert len(original_utterances) == len(generated_utterances) == len(speakers) == len(labels)

        for i in range(len(original_utterances)):
            utterance = original_utterances[i]
            generateds = generated_utterances[i]
            speaker = speakers[i]
            label = labels[i]
            # process the original utterance
            input_ids, attention_mask = process_sentence(
                utterance,
                max_seq_length,
                tokenizer,
                cls_token_at_end=cls_token_at_end,
                # xlnet has a cls token at the end
                cls_token=cls_token,
                cls_token_segment_id=cls_token_segment_id,
                sep_token=sep_token,
                sep_token_extra=sep_token_extra,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=pad_on_left,
                # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id
            )
            dialog_input_ids.append(input_ids)
            dialog_attention_mask.append(attention_mask)
            # process generated utterances
            d_g_ids = []
            d_g_mask = []
            for g in generateds:
                g_ids, g_mask = process_sentence(
                    g,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end=cls_token_at_end,
                    # xlnet has a cls token at the end
                    cls_token=cls_token,
                    cls_token_segment_id=cls_token_segment_id,
                    sep_token=sep_token,
                    sep_token_extra=sep_token_extra,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=pad_on_left,
                    # pad on the left for xlnet
                    pad_token=pad_token,
                    pad_token_segment_id=pad_token_segment_id
                )
                d_g_ids.append(g_ids)
                d_g_mask.append(g_mask)
            dialog_generated_input_ids.append(d_g_ids)
            dialog_generated_attention_mask.append(d_g_mask)
            # for speaker of each uttercance
            dialog_speakers.append(speaker)
            # for label_id of each utterance
            dialog_label_ids.append(data_processor.label2idx[label])
        
        all_original_input_ids.append(dialog_input_ids)
        all_original_attention_mask.append(dialog_attention_mask)
        all_generated_input_ids.append(dialog_generated_input_ids)
        all_generated_attention_mask.append(dialog_generated_attention_mask)
        all_speakers.append(dialog_speakers)
        all_label_ids.append(dialog_label_ids)
    
    return all_original_input_ids, all_original_attention_mask, all_generated_input_ids, all_generated_attention_mask, all_speakers, all_label_ids


def load_examples(args, data_processor, tokenizer, split):
    logger.info("Loading data from data_utils.py...")
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = data_processor.get_examples(split=split)
    all_original_input_ids, all_original_attention_mask, all_generated_input_ids, all_generated_attention_mask, all_speakers, all_label_ids = convert_examples_to_features(
        examples,
        data_processor,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
    )

    dataset = ERCDataset(all_original_input_ids, all_original_attention_mask, all_generated_input_ids, all_generated_attention_mask, all_speakers, all_label_ids)

    return dataset
