import torch
from torch import nn
import torch.functional as F
import math

# 添加命令行参数
import argparse
parser = argparse.ArgumentParser(description='Parser for embedding')
parser.add_argument('--gpu', type=int, default=1)
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
    