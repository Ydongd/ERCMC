from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from models.encoder_layer import EncoderLayer, RelativeEncoderLayer

class SinPositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(SinPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class VanillaEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
    
    def forward(self, src_emb, src_mask=None):
        # src_emb = [1, len, hid]
        src_emb = self.dropout(src_emb)
        src_emb = self.layer_norm(src_emb)
        for enc_layer in self.layer_stack:
            src_emb = enc_layer(src_emb, src_emb, src_emb, mask=src_mask)
        return src_emb

class SinEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, n_position=200):
        super().__init__()
        self.position_enc = SinPositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
    
    def forward(self, src_emb, src_mask=None):
        # src_emb = [1, len, hid]
        src_emb = self.dropout(self.position_enc(src_emb))
        src_emb = self.layer_norm(src_emb)
        for enc_layer in self.layer_stack:
            src_emb = enc_layer(src_emb, src_emb, src_emb, mask=src_mask)
        return src_emb

class TrainableEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, n_position=200):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.Tensor(1, n_position, d_model))
        nn.init.xavier_uniform_(self.position_embeddings)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_emb, src_mask=None):
        src_emb = src_emb + self.position_embeddings[:, :src_emb.size(1)]
        src_emb = self.dropout(src_emb)
        src_emb = self.layer_norm(src_emb)
        for enc_layer in self.layer_stack:
            src_emb = enc_layer(src_emb, src_emb, src_emb, mask=src_mask)
        return src_emb

class RelativeEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, max_relative_position=5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            RelativeEncoderLayer(d_model, d_inner, n_head, d_k, d_v, max_relative_position=max_relative_position, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
    
    def forward(self, src_emb, src_mask=None):
        src_emb = self.dropout(src_emb)
        src_emb = self.layer_norm(src_emb)
        for enc_layer in self.layer_stack:
            src_emb = enc_layer(src_emb, src_emb, src_emb, mask=src_mask)
        return src_emb
