import torch
from torch import nn
import torch.functional as F
import math
from 手撕多头注意力机制.手写多头注意力计算 import MultiHeadAttention


# 3、Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)                 # 求均值， -1指倒数第一维
        var = x.var(-1, unbiased=False, keepdim=True)   # 求方差
        out = (x - mean) / torch.sqrt(var + self.eps)
        
        # 在训练过程中，self.gamma 和 self.beta 会逐渐学习到合适的值，从而调整 out 的分布
        out = self.gamma * out + self.beta
        return out
    

# 4、FFN
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DeconderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DeconderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(n_head, d_model)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.cross_attention = MultiHeadAttention(n_head, d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
        
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        # 下三角掩码，不能看到未来信息
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        if enc is not None:
            _x = x
            # 多出来填充部分的掩码
            x = self.cross_attention(x, enc, enc, s_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)
            
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x