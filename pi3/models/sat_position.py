import torch
import torch.nn as nn


class FourierEmbedder(nn.Module):
    def __init__(self, in_dim=2, embed_dim=1024, num_freqs=64, scale=10.0, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        
        # 随机初始化高斯矩阵 B，且不可训练 (fixed)
        # scale 决定了频率分布的带宽，值越大能捕获越细微的位置变化
        self.register_buffer("B", torch.randn(in_dim, num_freqs) * scale)
        
        # 计算输出维度
        self.out_dim = num_freqs * 2 + (in_dim if include_input else 0)
        
        # 最后的投影层，将频率特征映射到 Transformer 的维度
        self.proj = nn.Linear(self.out_dim, embed_dim)
        
        # 保持你原始代码的 Zero-init 策略
        # 这样在训练开始时，位置编码产生的扰动为 0，不会破坏预训练特征
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        # x: [Batch, ..., 2] (normalized coords)
        
        # 1. 投影到频率基: (2*pi * x @ B)
        x_proj = (2 * torch.pi * x) @ self.B
        
        # 2. 计算 sin 和 cos
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # 3. (可选) 拼接原始坐标
        if self.include_input:
            embed = torch.cat([x, embed], dim=-1)
            
        # 4. 投影回目标维度
        return self.proj(embed)