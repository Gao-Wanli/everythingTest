import torch
from torch import nn
import torch.functional as F
import math

# 添加命令行参数
import argparse
parser = argparse.ArgumentParser(description="Parser for MultiHeadAttention")
parser.add_argument('--gpu', type=int, default=1)
# 解析命令行参数
args = parser.parse_args()
torch.cuda.set_device(args.gpu)

# 具体实现
# 生成测试数据 (batchsize, time, dimension)
X = torch.randn(128, 64, 512)

# 多头注意力的参数
model_dim = 512
head_n = 8

class MultiHeadAttention(nn.Module):
    def __init__(self, head_n, model_dim):
        super(MultiHeadAttention, self).__init__()
        
        self.head_n = head_n
        self.head_d = model_dim
        # 每一头的维度
        self.att_dim = self.head_d // self.head_n     
        
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_combine = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v):
        batch_size, time, dimension = q.shape
        
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        # 维度划分
        q = q.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, time, self.head_n, self.att_dim).permute(0, 2, 1, 3)
        
        score = q @ k.transpose(2, 3) / math.sqrt(self.att_dim)
        # 生成下三角矩阵，相当于mask掉上三角
        mask = torch.tril(torch.ones(time, time))
        score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score)

        output = score @ v
        # .contiguous()让整个序列在内存中连续，只有这样才能执行.view()
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, time, dimension)
        
        output = self.W_combine(output)
        return output
    

attention = MultiHeadAttention(head_n, model_dim)
out = attention(X, X, X)
print(out, out.shape)
        
        
        