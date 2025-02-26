import torch
from torch import nn
import torch.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.head_d = d_model
        # 每一头的维度
        self.att_dim = self.head_d // self.n_head     
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None):
        batch_size, time, dimension = q.shape
        
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 维度划分
        q = q.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        
        score = q @ k.transpose(2, 3) / math.sqrt(self.att_dim)
        
        if mask is not None:
            # mask = torch.tril(torch.ones(time, time))
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score)

        output = score @ v
        # .contiguous()让整个序列在内存中连续，只有这样才能执行.view()
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, time, dimension)
        
        output = self.W_combine(output)
        return output
    
        
        
        