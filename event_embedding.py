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
    def __init__(self, P: int, H: int, W: int, d_model: int, norm: bool, return_embed_matrix: bool, norm_type:str='ln', activation:str='relu'):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        if norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif norm_type == 'rms':
            norm_class = nn.RMSNorm

        self.embed = nn.Sequential(
            nn.Linear(3, d_model // 4, bias=False),
            norm_class(d_model // 4),
            utils.create_activation(activation, inplace=True),
            nn.Linear(d_model // 4, d_model // 2, bias=False),
            norm_class(d_model // 2),
            utils.create_activation(activation, inplace=True),

            nn.Linear(d_model // 2, d_model, bias=False),
            norm_class(d_model),
        )
        self.norm = norm
        self.return_embed_matrix = return_embed_matrix

    def embedding_matrix(self, device, padding:bool):
        c = torch.arange(self.P * self.H * self.W, device=device, dtype=torch.float)
        x = c % self.W
        y = (c // self.W) % self.H
        p = c // (self.W * self.H)
        # assert torch.all(p * self.H * self.W + y * self.W + x == c)
        if self.norm:
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
        if self.return_embed_matrix:
            return self.embed_forward(p, y, x, valid_mask)

        else:
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

        if self.norm:
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
