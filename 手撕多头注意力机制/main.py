import torch
from torch import nn
import torch.functional as F
import math
from 手撕多头注意力机制.手写多头注意力计算 import MultiHeadAttention
from 手撕多头注意力机制.手写编码模块 import EncoderLayer, TransformerEmbedding
from 手撕多头注意力机制.手写解码模块 import DeconderLayer

class Encoder(nn.Module):
    def __init__(self, env_voc_size, max_len, d_model, fnn_hidden, n_head, n_layer, drop_prob, device):
        super(Encoder, self).__init__()
        
        self.embedding = TransformerEmbedding(d_model, max_len, env_voc_size, drop_prob, device)
        
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, fnn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )
        
    def forward(self, x, s_mask):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, s_mask)
        
        return x
    

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, fnn_hidden, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()
        
        self.embedding = TransformerEmbedding(d_model, max_len, dec_voc_size, drop_prob, device)
        
        self.layers = nn.ModuleList(
            [DeconderLayer(d_model, fnn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )
        
        self.fc = nn.Linear(d_model, dec_voc_size)
        
    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        
        dec = self.fc(dec)
        return dec
    
    
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_head, fnn_hidden, n_layer, drop_prob, device):
        super(Transformer, self).__init__()
        
        # size：词的个数
        self.encoder = Encoder(enc_voc_size, max_len, d_model, fnn_hidden, n_head, n_layer, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, fnn_hidden, n_head, n_layer, drop_prob, device)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        
        # (batch, time, len_1, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        
        # (batch, time, len_q, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        
        mask = q & k    # 与运算
        return mask
    
    # trg:target
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx)
        trg_mask = self.make_casual_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        
        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output
        
        