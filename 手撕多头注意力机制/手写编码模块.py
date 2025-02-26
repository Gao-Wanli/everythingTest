import torch
from torch import nn
import torch.functional as F
import math
from 手撕多头注意力机制.手写多头注意力计算 import MultiHeadAttention

# 添加命令行参数
import argparse
parser = argparse.ArgumentParser(description='Parser for embedding')
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()
torch.cuda.set_device(args.gpu)

# 1、token embedding
# 用于将 token（通常是单词、子词或字符）映射到一个 embed_size 维度的向量空间中。
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        # vocab_size：词表大小，即可以索引的 token 数量。
        # embed_size：嵌入向量的维度。
        # padding_idx=1？：指定填充索引，即填充 token 的索引值。在序列数据中，填充 token 用于填充序列长度，使得所有序列具有相同的长度。
        # 通常填充索引为1后，1的会全初始化为0，梯度永远为0不会影响参数优化
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=1)
        
        
# 2、position embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        
        # maxlen, d_model指定维度；device指定gpu
        # self.encoding用于存储位置编码
        self.encoding = torch.zeros(maxlen, d_model, device)
        self.encoding.requires_grad = False
        
        # pos用于计算位置编码
        pos = torch.arange(0, maxlen, device)
        # 扩展到二维，形状是(maxlen, 1)，每一行是一个位置索引
        pos = pos.float().unsqueeze(1)
        # 频率索引？？？
        _2i = torch.arange(0, d_model, step=2, device=device)   # [0, 2, 4, 6, 8, ..., d_model-2]
        # 正弦和余弦函数：
        # 通过正弦和余弦函数的交替使用，可以为每个位置生成一个唯一的编码。
        # 由于正弦和余弦函数的周期性，位置编码可以捕捉到序列中不同位置之间的相对距离。
        
        # 频率变化：
        # 10000 ** (_2i / d_model)是一个频率因子，用于控制不同维度的频率变化
        # 不同维度的编码具有不同的频率，低频维度捕捉较长的序列关系，高频维度捕捉较短的序列关系。
        self.encoding[:, 0::2] = torch.sin(pos, (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos, (10000 ** (_2i / d_model)))
    
    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
  

# 词嵌入、位置嵌入实现embedding
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
      
    
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

# 5、Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_head, d_model)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)
        
    def forward(self, x, mask):
        # 残差连接
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)
        
        x = self.drop2(x)
        x = self.norm2(x + _x)
        
        return x