import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils



class FFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int,
                 activation: str, bias: bool, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.activation = utils.create_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src




class EncoderLayer(nn.Module):
    def __init__(
            self,
            attention: nn.Module,
            d_model: int,
            dim_feedforward: int,
            activation: str,
            bias: bool,
            ffn_dropout: float,
            norm_type='ln',
            pre_norm:bool=False
    ):
        super().__init__()
        self.attention = attention
        if norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif norm_type == 'rms':
            norm_class = nn.RMSNorm
        self.pre_norm = pre_norm
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)

        self.ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, dropout=ffn_dropout,
                        bias=bias)
        

    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor, drop_path: nn.Module):
        if self.pre_norm:
            src = src + drop_path(self._sa_block(self.norm1(src), valid_mask=valid_mask))
            if self.norm2 is not None:
                src = src + drop_path(self._ff_block(self.norm2(src)))
        else:
            src = self.norm1(src + drop_path(self._sa_block(src=src, valid_mask=valid_mask)))
            if self.norm2 is not None:
                src = self.norm2(src + drop_path(self._ff_block(src)))
        return src

    def _sa_block(
            self,
            src: torch.Tensor,
            valid_mask: torch.BoolTensor,
    ):
        return self.attention(src=src, valid_mask=valid_mask)

    def _ff_block(self, src):
        return self.ffn(src)
