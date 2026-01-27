from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import importlib
import utils


class Embedding(nn.Embedding):
    def forward_with_mask(self, indices: torch.LongTensor, valid_mask: torch.BoolTensor):
        # indices.shape = [B, L]
        # valid_mask.shape = [B, L]
        indices += 1
        # although the pad value is -1, they may change, e.g., after p * H * W + y * W + x
        indices *= valid_mask  # set padding to zeros
        return super().forward(indices)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.padding_idx is not None:
            assert self.padding_idx == 0
            self.fwd = self.forward_with_mask

        else:
            # become the original nn.Embedding
            self.fwd = utils.In2Out1st()

    def forward(self, indices: torch.LongTensor, valid_mask=None):
        return self.fwd(indices, valid_mask)


class EventEmbedding(Embedding):
    def __init__(self, P: int, H: int, W: int, d: int):
        self.P = P
        self.H = H
        self.W = W
        self.d = d
        super().__init__(num_embeddings=P * H * W + 1, embedding_dim=d, padding_idx=0)

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask=None):
        indices = p * (self.H * self.W) + y * self.W + x
        return super().forward(indices, valid_mask)


class MLPEmbedding(nn.Module):
    def __init__(self, d_model: int, norm_type:str='ln', activation:str='relu', in_features:int=3):
        super().__init__()

        if norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif norm_type == 'rms':
            norm_class = nn.RMSNorm


        self.embed = nn.Sequential(
            nn.Linear(in_features, d_model // 4, bias=False),
            norm_class(d_model // 4),
            utils.create_activation(activation),
            nn.Linear(d_model // 4, d_model // 2, bias=False),
            norm_class(d_model // 2),
            utils.create_activation(activation),

            nn.Linear(d_model // 2, d_model, bias=False),
            norm_class(d_model),
        )


    def forward(self, c, valid_mask: torch.BoolTensor):
        c *= valid_mask.float().unsqueeze(2)
        c = self.embed(c)
        return c


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        # 随机初始化高斯矩阵 B，不可学习
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: [Batch, ..., input_dim]
        # x_proj: [Batch, ..., mapping_size]
        x_proj = (2. * torch.pi * x) @ self.B
        # 拼接 sin 和 cos，输出维度变为 2 * mapping_size
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierMLPEmbedding(nn.Module):
    def __init__(self, d_model, activation='gelu', in_features:int=3):
        super().__init__()
        
        # 1. 傅里叶特征映射
        # 将 3维坐标 (x,y,p) 映射到更宽的空间，例如 d_model 的一半
        # 输出维度是 2 * (d_model // 2) = d_model
        mapping_dim = d_model // 8
        self.fourier_map = FourierFeatureMapping(input_dim=in_features, mapping_size=mapping_dim, scale=10)
        
        # 2. MLP 主体
        # 输入维度现在是 d_model (由傅里叶特征得来)
        # 结构改为：D -> D -> D (不再有 D/4 的瓶颈)
        self.mlp = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 4, bias=False),
            nn.LayerNorm(d_model // 4),
            self._get_act(activation),
            
            nn.Linear(d_model // 4, d_model // 2, bias=False),
            nn.LayerNorm(d_model // 2),
            self._get_act(activation),
            
            # 最后一层可以不加激活，或者保持原样
            nn.Linear(d_model // 2, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

    def _get_act(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            return nn.ReLU()

    def forward(self, x):
        # x: [B, L, 3]
        x = self.fourier_map(x) # -> [B, L, D]
        x = self.mlp(x)         # -> [B, L, D]
        return x



class GatedSpatioTemporalFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        # 1. Intensity 投影层：将标量强度映射为 d_model 维向量
        self.intensity_proj = nn.Linear(1, d_model)
        self.norm_intensity = nn.LayerNorm(d_model)

        # 2. 门控网络：决定融合比例
        # 输入变为 3 * d_model (Spatial + Temporal + Intensity)
        # Intensity 也应该参与决策，比如强度很大时，可能更需要关注 Spatial
        self.gate_net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Sigmoid() 
        )
        
        self.out_proj = nn.Identity()

    def forward(self, v_s, v_t, rho=None):
        """
        v_s: Spatial Embedding [B, L, D]
        v_t: Temporal Embedding [B, L, D]
        rho: Intensity / Density [B, L] or [B, L, 1]
        """
        
        # --- Step 1: 处理 Intensity (独立特征) ---
        if rho is not None:
            if rho.dim() == 2:
                rho = rho.unsqueeze(-1)
            
            # log1p + Linear 投影，代替原本的乘法缩放
            rho_log = torch.log1p(rho.float())
            v_rho = self.intensity_proj(rho_log)
            v_rho = self.norm_intensity(v_rho)
        else:
            # 如果没有强度信息，就用 0 填充，不影响加法
            v_rho = torch.zeros_like(v_s)

        # --- Step 2: 计算门控系数 (Gate) ---
        # 拼接所有信息，让网络由 S, T, I 共同决定关注点
        combined = torch.cat([v_s, v_t, v_rho], dim=-1) # [B, L, 3*D]
        
        # 计算权重 alpha
        alpha = self.gate_net(combined)
        
        # --- Step 3: 融合 (修改点) ---
        # A. 加权融合时空特征
        fused_st = alpha * v_s + (1 - alpha) * v_t
        
        # B. 全局加上强度特征 (Global Additive)
        # 这等价于 "BaseFeature + IntensityBonus"
        # 既保留了 (S+T) 的结构，又实现了 Intensity 对整体的增强
        v_out = fused_st + v_rho
        
        return self.out_proj(v_out)