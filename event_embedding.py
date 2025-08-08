from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import importlib

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1)

class ParametricEmbedding(nn.Module):
    def __init__(self, P: int, H: int, W: int, d_model: int, return_embed_matrix: bool):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W

        self.embed = nn.Sequential(
            nn.Linear(3, d_model // 4, bias=False),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model // 2, bias=False),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),

            nn.Linear(d_model // 2, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        self.return_embed_matrix = return_embed_matrix

    def embedding_matrix(self, device, padding:bool):
        c = torch.arange(self.P * self.H * self.W, device=device, dtype=torch.float)
        x = c % self.W
        y = (c // self.W) % self.H
        p = c // (self.W * self.H)
        # assert torch.all(p * self.H * self.W + y * self.W + x == c)

        if self.P > 1:
            p *= (2. / (self.P - 1))
            p -= 1.
        if self.H > 1:
            y *= (2. / (self.H - 1))
            y -= 1.
        if self.W > 1:
            x *= (2. / (self.W - 1))
            x -= 1.

        w = self.embed(torch.stack((x, y, p), dim=1))
        if not padding:
            return w  # [PHW, d]

        return torch.cat((
            torch.zeros([1, w.shape[1]], device=w.device, dtype=w.dtype),
            w
        ), dim=0)  # [PHW + 1, d]

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask: torch.BoolTensor):
        return self.direct_forward(p, y, x, valid_mask)


    def embed_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask: torch.BoolTensor):
        w = self.embedding_matrix(x.device, padding=True)
        index = p * (self.H * self.W) + y * self.W + x
        index += 1
        index *= valid_mask.long()
        if self.return_embed_matrix:
            return F.embedding(index, w, padding_idx=0), w
        else:
            return F.embedding(index, w, padding_idx=0)


    def direct_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask: torch.BoolTensor):
        p = p.float()
        y = y.float()
        x = x.float()

        valid_mask_float = valid_mask.float()


        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.

        c = torch.stack((x, y, p), dim=2)
        c *= valid_mask_float.unsqueeze(2)

        c = self.embed(c)
        return c

class TimestampConv(nn.Module):
    def __init__(self, kernel_size:int, d:int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            Transpose(1, 2),
            nn.LayerNorm(d // 4),
            Transpose(1, 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(d // 4, d // 2, groups=d // 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            Transpose(1, 2),
            nn.LayerNorm(d // 2),
            Transpose(1, 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(d // 2, d, groups=d // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            Transpose(1, 2),
            nn.LayerNorm(d),
        )

    def forward(self, t: torch.Tensor):
        # shape = [B, L]
        assert t.dim() == 2
        t = t.unsqueeze(1)  # [B, 1, L]
        t = self.conv(t)  # [B, L, d]

        return t