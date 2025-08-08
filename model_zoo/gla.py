import torch
import torch.nn as nn
import torch.nn.functional as F
import fla
from . import common, bi_gla

class GatedLinearAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, conv_size:int, bidirectional:bool, **kwargs):
        super().__init__()
        if bidirectional:
            self.attention = bi_gla.BiGatedLinearAttention(hidden_size=d_model, num_heads=num_heads, use_short_conv=conv_size > 0, conv_size=conv_size, **kwargs)
        else:
            self.attention = fla.layers.GatedLinearAttention(hidden_size=d_model, num_heads=num_heads, use_short_conv=conv_size > 0, conv_size=conv_size, **kwargs)
        # triton要求维度最低是16，否则会报错


    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor):
        return self.attention(src, attention_mask=valid_mask)[0]

class GLAEncoderLayer(common.EncoderLayer):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dim_feedforward: int,
                 activation: str,
                 bias: bool,
                 bidirectional:bool,
                 conv_size:int,
                 ffn_type:str,
                 ffn_dropout:float,
                 **kwargs
                 ):
        self.d_model = d_model
        super().__init__(attention=GatedLinearAttention(d_model=d_model, num_heads=num_heads, bidirectional=bidirectional, conv_size=conv_size, **kwargs), d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, bias=bias, ffn_type=ffn_type, ffn_dropout=ffn_dropout)



class GLAEncoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            encoder_layer: nn.Module,
            norm: nn.Module,
    ) -> None:
        super().__init__()

        self.layers = common._get_clones(encoder_layer, n_layers)
        self.norm = norm


    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor):

        for m in self.layers:
            src = m(src=src, valid_mask=valid_mask)
        src = self.norm(src)
        return src

def create_gla_encoder(d_model: int, num_heads: int, d_feedforward: int, n_layers: int, activation:str, conv_size:int, ffn_type:str, ffn_dropout:float, bias:bool, bidirectional: bool, **kwargs):
    encoder_layer = GLAEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=d_feedforward, activation=activation, bidirectional=bidirectional, bias=bias, conv_size=conv_size, ffn_type=ffn_type, ffn_dropout=ffn_dropout, **kwargs)

    norm = nn.LayerNorm(d_model)
    return GLAEncoder(encoder_layer=encoder_layer, n_layers=n_layers, norm=norm)