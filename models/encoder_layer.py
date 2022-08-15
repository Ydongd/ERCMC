from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        assert d_model % n_head == 0
        assert d_k * n_head == d_model
        assert d_v * n_head == d_model
        
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).cuda()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # For multi-head self attention
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        # For position-wise feed forward
        self.w_1 = nn.Linear(d_model, d_inner) # position-wise
        self.w_2 = nn.Linear(d_inner, d_model) # position-wise
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # q: b x n x lq x dv
        # attn: b x n x lq x lq
        attn = torch.matmul(q / self.scale, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        output = self.layer_norm(output)

        # Position-wise feed forward
        residual = output
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output += residual
        output = self.layer_norm(output)

        return output


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class RelativeEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, max_relative_position, dropout=0.1):
        super(RelativeEncoderLayer, self).__init__()
        assert d_model % n_head == 0
        assert d_k * n_head == d_model
        assert d_v * n_head == d_model

        self.scale = torch.sqrt(torch.FloatTensor([d_k])).cuda()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # For multi-head self attention
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)
        # For position-wise feed forward
        self.w_1 = nn.Linear(d_model, d_inner) # position-wise
        self.w_2 = nn.Linear(d_inner, d_model) # position-wise
        # For relative position embedding
        self.relative_position_k = RelativePosition(self.d_k, max_relative_position)
        self.relative_position_v = RelativePosition(self.d_v, max_relative_position)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # Transpose for attention dot product: b x n x lq x dv
        q1, k1 = q.view(sz_b, len_q, n_head, d_k).transpose(1, 2), k.view(sz_b, len_k, n_head, d_k).transpose(1, 2)

        attn1 = torch.matmul(q1, k1.transpose(2, 3))

        q2 = q.permute(1, 0, 2).contiguous().view(len_q, sz_b*self.n_head, self.d_k)
        k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(q2, k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(sz_b, self.n_head, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [b, n, lq, lk]
        v1 = v.view(sz_b, -1, self.n_head, d_v).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, v1)
        v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, sz_b*self.n_head, len_k)
        weight2 = torch.matmul(weight2, v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(sz_b, self.n_head, len_q, self.d_v)
        
        # output = [b, n, lq, d_v]
        output = weight1 + weight2

        # output = [b, lq, n, d_v]
        output = output.permute(0, 2, 1, 3).contiguous()
        
        # output = [b, lq, d_model]
        output = output.view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)

        # Position-wise feed forward
        residual = output
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)
        output += residual
        output = self.layer_norm(output)

        return output
