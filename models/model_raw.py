from logging import log
from turtle import forward
import torch
import torch.nn as nn
from arguments import get_model_classes, get_args
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import re
from models.CRF import CRF
from models.encoder import VanillaEncoder, SinEncoder, TrainableEncoder, RelativeEncoder

class Model(nn.Module):
    def __init__(self, args, tokenizer, position_mode, window, d_inner=2048, n_layers=2, n_head=8, dropout=0.1, n_position=120):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.dropout_prob)
        self.window = window

        self.plm = model_config['model'].from_pretrained(
            args.model_name_or_path
        )

        self.fc = nn.Linear(self.plm.config.hidden_size, self.num_labels)

    def forward(self, batch_dialog_input_ids, batch_dialog_attention_mask, batch_dialog_generated_input_ids, batch_dialog_generated_attention_mask, batch_dialog_speakers, batch_dialog_label_ids, mode):
        batch_size = len(batch_dialog_label_ids)
        all_logits = []
        for j in range(batch_size):
            dialog_input_ids = batch_dialog_input_ids[j]
            dialog_attention_mask = batch_dialog_attention_mask[j]
            dialog_speakers = batch_dialog_speakers[j]
            dialog_label_ids = batch_dialog_label_ids[j]

            len_d = len(dialog_label_ids)

            dialog_input_ids = torch.tensor(dialog_input_ids).cuda()
            dialog_attention_mask = torch.tensor(dialog_attention_mask).cuda()
            dialog_label_ids = torch.tensor(dialog_label_ids).cuda()

            # [u_num, hidden]
            dialog_embeds = self.plm(input_ids=dialog_input_ids, attention_mask=dialog_attention_mask)['last_hidden_state'][:,0]
            
            sequence_output = self.dropout(dialog_embeds)
            logits=self.fc(sequence_output)
            all_logits.append(logits)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, dialog_label_ids)
            if j == 0:
                all_loss = loss
            else:
                all_loss += loss

        if mode == "train":
            return all_loss
        else: # test
            return all_loss, all_logits, batch_dialog_label_ids


class ModelCRF(nn.Module):
    def __init__(self, args, tokenizer, position_mode, window, d_inner=2048, n_layers=2, n_head=8, dropout=0.1, n_position=120):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.dropout_prob)
        self.window = window
        self.crf = CRF(num_tags=args.num_labels, batch_first=True)

        self.plm = model_config['model'].from_pretrained(
            args.model_name_or_path
        )

        self.fc = nn.Linear(self.plm.config.hidden_size, self.num_labels)

    def forward(self, batch_dialog_input_ids, batch_dialog_attention_mask, batch_dialog_generated_input_ids, batch_dialog_generated_attention_mask, batch_dialog_speakers, batch_dialog_label_ids, mode):
        batch_size = len(batch_dialog_label_ids)
        all_logits = []
        for j in range(batch_size):
            dialog_input_ids = batch_dialog_input_ids[j]
            dialog_attention_mask = batch_dialog_attention_mask[j]
            dialog_speakers = batch_dialog_speakers[j]
            dialog_label_ids = batch_dialog_label_ids[j]

            len_d = len(dialog_label_ids)

            dialog_input_ids = torch.tensor(dialog_input_ids).cuda()
            dialog_attention_mask = torch.tensor(dialog_attention_mask).cuda()
            dialog_label_ids = torch.tensor(dialog_label_ids).cuda()

            # [u_num, hidden]
            dialog_embeds = self.plm(input_ids=dialog_input_ids, attention_mask=dialog_attention_mask)['last_hidden_state'][:,0]
            
            sequence_output = self.dropout(dialog_embeds)
            logits=self.fc(sequence_output)
            logits = logits.unsqueeze(0)
            dialog_label_ids = dialog_label_ids.unsqueeze(0)
            labels = torch.where(dialog_label_ids >= 0, dialog_label_ids, torch.zeros_like(dialog_label_ids))
            loss = self.crf(emissions=logits, tags=labels)
            loss = -1 * loss
            if mode == "test":
                logits = self.crf.decode(logits)
                all_logits.append(logits.squeeze(0))
            if j == 0:
                all_loss = loss
            else:
                all_loss += loss

        if mode == "train":
            return all_loss
        else: # test
            return all_loss, all_logits, batch_dialog_label_ids
