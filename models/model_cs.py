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

        self.fc = nn.Linear(self.plm.config.hidden_size * 3, self.num_labels)
        self.W_1 = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.W_2 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.W_3 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.W_4 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.x2h_c = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.h2h_c = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.x2h_s = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.h2h_s = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)

        d_model = self.plm.config.hidden_size
        assert d_model % n_head == 0
        d_k = int(d_model / n_head)
        d_v = int(d_model / n_head)
        if position_mode == "vanilla":
            self.encoder = VanillaEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)
        elif position_mode == "sin":
            self.encoder = SinEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)
        elif position_mode == "trainable":
            self.encoder = TrainableEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)
        elif position_mode == "relative":
            self.encoder = RelativeEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, max_relative_position=self.window)
        else:
            print("Wrong Position Mode!")
            exit()

    def forward(self, batch_dialog_input_ids, batch_dialog_attention_mask, batch_dialog_generated_input_ids, batch_dialog_generated_attention_mask, batch_dialog_speakers, batch_dialog_label_ids, mode):
        batch_size = len(batch_dialog_label_ids)
        all_logits = []
        for j in range(batch_size):
            dialog_input_ids = batch_dialog_input_ids[j]
            dialog_attention_mask = batch_dialog_attention_mask[j]
            dialog_generated_input_ids = batch_dialog_generated_input_ids[j]
            dialog_generated_attention_mask = batch_dialog_generated_attention_mask[j]
            dialog_speakers = batch_dialog_speakers[j]
            dialog_label_ids = batch_dialog_label_ids[j]

            len_d = len(dialog_label_ids)
            # Construct a mapping from speakers to utterances
            s2u = {}
            for i, speaker in enumerate(dialog_speakers):
                if speaker not in s2u:
                    s2u[speaker] = []
                s2u[speaker].append(i)

            dialog_input_ids = torch.tensor(dialog_input_ids).cuda()
            dialog_attention_mask = torch.tensor(dialog_attention_mask).cuda()
            dialog_label_ids = torch.tensor(dialog_label_ids).cuda()

            # [u_num, hidden]
            dialog_embeds = self.plm(input_ids=dialog_input_ids, attention_mask=dialog_attention_mask)['last_hidden_state'][:,0]

            sequence_output = torch.tensor([]).cuda()
            hx_c = torch.zeros((1, self.plm.config.hidden_size)).cuda()
            hx_s = torch.zeros((1, self.plm.config.hidden_size)).cuda()

            # Parsing every utterance to get multi-aware embeddings
            for i in range(len_d):
                # Context-aware embedding
                begin = max(i-self.window, 0)
                end = i + 1
                now_c = -1
                context_embeds = dialog_embeds[begin:end].unsqueeze(0)
                # Speaker-aware embedding
                s = dialog_speakers[i]
                us = s2u[s]
                index = us.index(i)
                begin = max(index-self.window, 0)
                end = index + 1
                s_index = us[begin:end]
                now_s = -1
                speaker_embeds = dialog_embeds[s_index].unsqueeze(0)
                # Encoding
                # [1, u_num, hidden]
                out_context = self.encoder(context_embeds)
                out_speaker = self.encoder(speaker_embeds)
                # Obtain embedding corresponding to i
                # [1, hidden]
                out_c = out_context[:, now_c]
                out_s = out_speaker[:, now_s]
                # Obtain local embedding
                origin_c = out_c # [1, hidden]
                origin_s = out_s
                local_c = out_context[:, :-1].squeeze(0) # [u_num, hidden]
                local_s = out_speaker[:, :-1].squeeze(0)
                local_c_ = self.W_1(local_c)
                local_s_ = self.W_1(local_s)
                weight_c = F.softmax(self.tanh(torch.mm(local_c_, origin_c.transpose(0, 1))), dim=0) # [u_num, 1]
                weight_s = F.softmax(self.tanh(torch.mm(local_s_, origin_s.transpose(0, 1))), dim=0)
                state_c = torch.sum(torch.mul(weight_c, local_c), dim=0).unsqueeze(0) # [1, hidden]
                state_s = torch.sum(torch.mul(weight_s, local_s), dim=0).unsqueeze(0)
                # Track the local state
                # For context
                x_t = self.x2h_c(state_c)
                h_t = self.h2h_c(hx_c)
                x_reset, x_upd, x_new = x_t.chunk(3, 1)
                h_reset, h_upd, h_new = h_t.chunk(3, 1)
                reset_gate = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))
                hx_c = update_gate * hx_c + (1 - update_gate) * new_gate
                # For speaker
                x_t = self.x2h_s(state_s)
                h_t = self.h2h_s(hx_s)
                x_reset, x_upd, x_new = x_t.chunk(3, 1)
                h_reset, h_upd, h_new = h_t.chunk(3, 1)
                reset_gate = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))
                hx_s = update_gate * hx_s + (1 - update_gate) * new_gate
                # Final
                out_f = self.W_2(torch.cat((out_c, out_s), dim=-1)) # [1, hidden]
                gs_f = self.W_3(torch.cat((state_c, state_s), dim=-1)) # [1,hidden]
                hx_f = self.W_4(torch.cat((hx_c, hx_s), dim=-1)) # [1,hidden]
                final = torch.cat((out_f, gs_f, hx_f), dim=-1) # [1, hidden * 3]

                sequence_output = torch.cat((sequence_output, final), dim=0)
            
            sequence_output = self.dropout(sequence_output)
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

        self.fc = nn.Linear(self.plm.config.hidden_size * 3, self.num_labels)
        self.W_1 = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size)
        self.W_2 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.W_3 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.W_4 = nn.Linear(self.plm.config.hidden_size * 2, self.plm.config.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.x2h_c = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.h2h_c = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.x2h_s = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)
        self.h2h_s = nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size * 3)

        d_model = self.plm.config.hidden_size
        assert d_model % n_head == 0
        d_k = int(d_model / n_head)
        d_v = int(d_model / n_head)
        if position_mode == "vanilla":
            self.encoder = VanillaEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout)
        elif position_mode == "sin":
            self.encoder = SinEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)
        elif position_mode == "trainable":
            self.encoder = TrainableEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position)
        elif position_mode == "relative":
            self.encoder = RelativeEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, max_relative_position=self.window)
        else:
            print("Wrong Position Mode!")
            exit()

    def forward(self, batch_dialog_input_ids, batch_dialog_attention_mask, batch_dialog_generated_input_ids, batch_dialog_generated_attention_mask, batch_dialog_speakers, batch_dialog_label_ids, mode):
        batch_size = len(batch_dialog_label_ids)
        all_logits = []
        for j in range(batch_size):
            dialog_input_ids = batch_dialog_input_ids[j]
            dialog_attention_mask = batch_dialog_attention_mask[j]
            dialog_generated_input_ids = batch_dialog_generated_input_ids[j]
            dialog_generated_attention_mask = batch_dialog_generated_attention_mask[j]
            dialog_speakers = batch_dialog_speakers[j]
            dialog_label_ids = batch_dialog_label_ids[j]

            len_d = len(dialog_label_ids)
            # Construct a mapping from speakers to utterances
            s2u = {}
            for i, speaker in enumerate(dialog_speakers):
                if speaker not in s2u:
                    s2u[speaker] = []
                s2u[speaker].append(i)

            dialog_input_ids = torch.tensor(dialog_input_ids).cuda()
            dialog_attention_mask = torch.tensor(dialog_attention_mask).cuda()
            dialog_label_ids = torch.tensor(dialog_label_ids).cuda()

            # [u_num, hidden]
            dialog_embeds = self.plm(input_ids=dialog_input_ids, attention_mask=dialog_attention_mask)['last_hidden_state'][:,0]

            sequence_output = torch.tensor([]).cuda()
            hx_c = torch.zeros((1, self.plm.config.hidden_size)).cuda()
            hx_s = torch.zeros((1, self.plm.config.hidden_size)).cuda()

            # Parsing every utterance to get multi-aware embeddings
            for i in range(len_d):
                # Context-aware embedding
                begin = max(i-self.window, 0)
                end = i + 1
                now_c = -1
                context_embeds = dialog_embeds[begin:end].unsqueeze(0)
                # Speaker-aware embedding
                s = dialog_speakers[i]
                us = s2u[s]
                index = us.index(i)
                begin = max(index-self.window, 0)
                end = index + 1
                s_index = us[begin:end]
                now_s = -1
                speaker_embeds = dialog_embeds[s_index].unsqueeze(0)
                # Encoding
                # [1, u_num, hidden]
                out_context = self.encoder(context_embeds)
                out_speaker = self.encoder(speaker_embeds)
                # Obtain embedding corresponding to i
                # [1, hidden]
                out_c = out_context[:, now_c]
                out_s = out_speaker[:, now_s]
                # Obtain local embedding
                origin_c = out_c # [1, hidden]
                origin_s = out_s
                local_c = out_context[:, :-1].squeeze(0) # [u_num, hidden]
                local_s = out_speaker[:, :-1].squeeze(0)
                local_c_ = self.W_1(local_c)
                local_s_ = self.W_1(local_s)
                weight_c = F.softmax(self.tanh(torch.mm(local_c_, origin_c.transpose(0, 1))), dim=0) # [u_num, 1]
                weight_s = F.softmax(self.tanh(torch.mm(local_s_, origin_s.transpose(0, 1))), dim=0)
                state_c = torch.sum(torch.mul(weight_c, local_c), dim=0).unsqueeze(0) # [1, hidden]
                state_s = torch.sum(torch.mul(weight_s, local_s), dim=0).unsqueeze(0)
                # Track the local state
                # For context
                x_t = self.x2h_c(state_c)
                h_t = self.h2h_c(hx_c)
                x_reset, x_upd, x_new = x_t.chunk(3, 1)
                h_reset, h_upd, h_new = h_t.chunk(3, 1)
                reset_gate = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))
                hx_c = update_gate * hx_c + (1 - update_gate) * new_gate
                # For speaker
                x_t = self.x2h_s(state_s)
                h_t = self.h2h_s(hx_s)
                x_reset, x_upd, x_new = x_t.chunk(3, 1)
                h_reset, h_upd, h_new = h_t.chunk(3, 1)
                reset_gate = torch.sigmoid(x_reset + h_reset)
                update_gate = torch.sigmoid(x_upd + h_upd)
                new_gate = torch.tanh(x_new + (reset_gate * h_new))
                hx_s = update_gate * hx_s + (1 - update_gate) * new_gate
                # Final
                out_f = self.W_2(torch.cat((out_c, out_s), dim=-1)) # [1, hidden]
                gs_f = self.W_3(torch.cat((state_c, state_s), dim=-1)) # [1,hidden]
                hx_f = self.W_4(torch.cat((hx_c, hx_s), dim=-1)) # [1,hidden]
                final = torch.cat((out_f, gs_f, hx_f), dim=-1) # [1, hidden * 3]

                sequence_output = torch.cat((sequence_output, final), dim=0)
            
            sequence_output = self.dropout(sequence_output)
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
