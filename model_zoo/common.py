import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from fla.modules.mlp import GatedMLP


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def create_activation(act: str, inplace:bool=False):
    if act == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU(inplace=inplace)
    elif act == 'sigmoid':
        if inplace:
            return torch.sigmoid_
        else:
            return nn.Sigmoid()

    else:
        raise NotImplementedError(act)

class FFN(nn.Module):
    def __init__(self, d_model:int, dim_feedforward: int,
        activation: str, bias:bool, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.activation = create_activation(activation)
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
        bias:bool,
        ffn_type: str,
        ffn_dropout:float
    ):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if ffn_type == 'plain':
            self.ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, dropout=ffn_dropout, bias=bias)
        elif ffn_type == 'gated_mlp':
            assert ffn_dropout == 0.
            assert not bias
            self.ffn = GatedMLP(hidden_size=d_model, intermediate_size=dim_feedforward, hidden_act=activation)

        elif ffn_type == 'none':
            assert ffn_dropout == 0.
            assert not bias
            self.ffn = None
            del self.norm2
            self.norm2 = None
        else:
            raise NotImplementedError(ffn_type)

    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor):

        src = self.norm1(src + self._sa_block(src=src, valid_mask=valid_mask))
        if self.norm2 is not None:
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(
        self,
        src: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ):
        return self.attention(src=src, valid_mask=valid_mask)


    def _ff_block(self, src):
        return self.ffn(src)

def sequence_avg_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride:int, avg:bool):
    B, L, d = tokens.shape
    if L % stride != 0:
        print(tokens.shape, stride)
        exit(-1)
    valid_mask = valid_mask.view(B, L // stride, stride).float()
    tokens = tokens.view(B, L // stride, stride, d)
    valid_mask_sum = valid_mask.sum(2)
    if avg:
        tokens = (tokens * valid_mask.unsqueeze(3)).sum(2) / (valid_mask_sum.unsqueeze(2) + 1e-5)
    else:
        tokens = (tokens * valid_mask.unsqueeze(3)).sum(2)
    valid_mask = valid_mask_sum > 0
    return tokens, valid_mask

def sequence_cat_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride:int):
    B, L, d = tokens.shape
    assert L % stride == 0
    tokens = (tokens * valid_mask.to(tokens).unsqueeze(2)).view(B, L // stride, stride * d)
    valid_mask = valid_mask.view(B, L // stride, stride).float()
    valid_mask_sum = valid_mask.sum(2)
    valid_mask = valid_mask_sum > 0
    return tokens, valid_mask


def sequence_max_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride:int):
    B, L, d = tokens.shape
    if L % stride != 0:
        print(tokens.shape, stride)
        exit(-1)
    tokens = tokens.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
    valid_mask = valid_mask.view(B, L // stride, stride).float()
    tokens = tokens.view(B, L // stride, stride, d)
    tokens = tokens.max(2)[0]
    valid_mask = valid_mask.sum(2) > 0
    return tokens, valid_mask

