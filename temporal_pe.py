import torch
import torch.nn as nn

import utils

class SinTemporalPositionalEncoding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

        _2j = torch.arange(0, d, step=2)
        denominator = 10000 ** (_2j / d).view(1, 1, d // 2)
        self.register_buffer('denominator', denominator)
        self.scale = nn.Parameter(torch.empty(1))
        self.scale.data.fill_(1000)

    def forward(self, t):
        # t.shape = [B, L]
        '''
        encoding.shape = [1, L, d]
        encoding[:, i, 2j] = sin(t[:, i] / 10000^{2j / d})
        encoding[:, i, 2j+1] = cos(t[:, i] / 10000^{2j / d})

        '''
        B, L = t.shape

        t = t * self.scale

        encoding = torch.zeros([B, L, self.d], device=t.device, dtype=t.dtype)
        pos = t.unsqueeze(2) / self.denominator

        encoding[:, :, 0::2] = torch.sin(pos)
        encoding[:, :, 1::2] = torch.cos(pos)
        return encoding

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # 这里的参数形状设为 [1, C, 1] 以便直接广播，无需 transpose
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x shape: [B, C, L]
        # 在通道维度 (dim=1) 计算均值和方差
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm

class RMSNorm1d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))

    def forward(self, x):
        # x shape: [B, C, L]
        # RMSNorm 不需要减均值，只除以均方根
        var = x.pow(2).mean(dim=1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight
    


class Conv(nn.Module):
    def __init__(self, kernel_size:int, d:int, norm_type:str='ln', activation:str='relu'):
        super().__init__()
        if norm_type == 'ln':
            norm_class = LayerNorm1d
        elif norm_type == 'rms':
            norm_class = RMSNorm1d

        self.conv = nn.Sequential(
            nn.Conv1d(1, d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            norm_class(d // 4),
            utils.create_activation(activation),

            nn.Conv1d(d // 4, d // 2, groups=d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            norm_class(d // 2),
            utils.create_activation(activation),

            nn.Conv1d(d // 2, d, groups=d // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            norm_class(d),
        )

    def forward(self, t: torch.Tensor):
        # shape = [B, L]
        assert t.dim() == 2
        t = t.unsqueeze(1)  # [B, 1, L]
        t = self.conv(t)  # [B, d, L]
        t = t.transpose(1, 2)

        return t



class Conv_(nn.Module):
    def __init__(self, kernel_size:int, d:int, norm_type:str='ln', activation:str='relu'):
        super().__init__()
        if norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif norm_type == 'rms':
            norm_class = nn.RMSNorm

        self.conv = nn.Sequential(
            nn.Conv1d(1, d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            utils.Transpose(1, 2),
            norm_class(d // 4),
            utils.Transpose(1, 2),
            utils.create_activation(activation, inplace=True),

            nn.Conv1d(d // 4, d // 2, groups=d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            utils.Transpose(1, 2),
            norm_class(d // 2),
            utils.Transpose(1, 2),
            utils.create_activation(activation, inplace=True),

            nn.Conv1d(d // 2, d, groups=d // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            utils.Transpose(1, 2),
            norm_class(d),
        )

    def forward(self, t: torch.Tensor):
        # shape = [B, L]
        assert t.dim() == 2
        t = t.unsqueeze(1)  # [B, 1, L]
        t = self.conv(t)  # [B, L, d]

        return t

class FourierTimeEmbedding(nn.Module):
    def __init__(self, output_dim, scale=100.0):
        super().__init__()
        self.output_dim = output_dim
        # 随机初始化频率矩阵，不可学习，或者设为可学习
        # 注意：时间差通常很小（微秒级归一化后），scale 需要大一点来捕捉高频
        self.register_buffer('freqs', torch.randn(1, output_dim // 2) * scale)

    def forward(self, t):
        # t: [B, 1, L]
        # output: [B, dim, L]
        
        # 调整形状进行广播: [B, 1, L] -> [B, L, 1]
        t = t.transpose(1, 2) 
        
        # 投影: [B, L, 1] @ [1, dim/2] -> [B, L, dim/2]
        args = t @ self.freqs
        
        # cat sin, cos -> [B, L, dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # 变回 Conv1d 需要的形状: [B, dim, L]
        return embedding.transpose(1, 2)

class FourierTemporalEmbedding(nn.Module):
    def __init__(self, kernel_size: int, d: int, norm_type: str = 'ln', activation: str = 'gelu'):
        super().__init__()
        
        # 1. 傅里叶特征映射: 1 -> d
        # 直接映射到目标维度 d，这比逐渐升维 (d/4 -> d/2 -> d) 更能保留信息
        self.fourier = FourierTimeEmbedding(d, scale=50.0)
        
        # 2. 特征变换与时序聚合 (类似 MobileNetV2 Block / Conformer)
        # 包含: Pointwise (混合通道) -> Depthwise (聚合时序) -> Pointwise (混合通道)
        
        if norm_type == 'ln':
            # LayerNorm 通常对 [B, L, D] 操作，但在 Conv 中我们通常处理 [B, D, L]
            # 为了方便，这里用 GroupNorm 替代 LayerNorm (GN with 1 group == LN on channel)
            # 或者坚持用 transpose + LN
            norm_layer = lambda dim: nn.GroupNorm(1, dim) 
        else:
            norm_layer = nn.BatchNorm1d

        self.mlp_conv = nn.Sequential(
            # A. 第一次投影: 加强特征交互
            nn.Conv1d(d, d, kernel_size=1),
            norm_layer(d),
            utils.create_activation(activation),
            
            # B. Depthwise Conv: 提取时序上下文 (Time Mixing)
            # 这一步只看邻居，不改变通道特征
            nn.Conv1d(d, d, kernel_size=kernel_size, stride=1, 
                      padding=(kernel_size - 1) // 2, groups=d, bias=False),
            norm_layer(d),
            utils.create_activation(activation),
            
            # C. Pointwise Conv: 再次混合通道
            nn.Conv1d(d, d, kernel_size=1),
            norm_layer(d)
        )

    def forward(self, t: torch.Tensor):
        # t shape = [B, L]
        
        # 1. 维度调整 [B, 1, L]
        if t.dim() == 2:
            t = t.unsqueeze(1)
            
        # 2. 傅里叶编码 [B, 1, L] -> [B, d, L]
        x = self.fourier(t)
        
        # 3. 卷积处理
        x = self.mlp_conv(x)
        
        # 4. 转置为 Transformer 需要的 [B, L, d]
        x = x.transpose(1, 2)
        
        return x