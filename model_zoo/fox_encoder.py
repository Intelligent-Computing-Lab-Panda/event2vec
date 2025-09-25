import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo import common, fox
import utils


class FoxLinearAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__()
               
        self.attention = fox.BiForgettingAttention(hidden_size=d_model, num_heads=num_heads, num_kv_heads=max(num_heads // 2, 1), qk_norm=True, use_output_gate=False, **kwargs)

        # triton要求维度最低是16，否则会报错

    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor):
        return self.attention(src, attention_mask=valid_mask)[0]


class FoxEncoderLayer(common.EncoderLayer):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dim_feedforward: int,
                 activation: str,
                 bias: bool,
                 ffn_dropout: float,
                 norm_type:str,
                 pre_norm:bool=False,
                 **kwargs
                 ):
        self.d_model = d_model
        super().__init__(
            attention=FoxLinearAttention(d_model=d_model, num_heads=num_heads, **kwargs), d_model=d_model,
            dim_feedforward=dim_feedforward, activation=activation, bias=bias,
            ffn_dropout=ffn_dropout, norm_type=norm_type, pre_norm=pre_norm)


from timm.layers.drop import DropPath


class FoxEncoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            encoder_layer: nn.Module,
            norm: nn.Module,
            drop_path: float = 0.
    ) -> None:
        super().__init__()

        self.layers = utils._get_clones(encoder_layer, n_layers)
        self.norm = norm

        self.drop_path = nn.ModuleList()
        for i in range(n_layers):
            if drop_path > 0:
                self.drop_path.append(
                    DropPath(i / (n_layers - 1) * drop_path)
                )
            else:
                self.drop_path.append(nn.Identity())

    def forward(self, src: torch.Tensor, valid_mask: torch.BoolTensor):

        for i, m in enumerate(self.layers):
            src = m(src=src, valid_mask=valid_mask, drop_path=self.drop_path[i])
        src = self.norm(src)
        return src


def create_fox_encoder(d_model: int, num_heads: int, d_feedforward: int, n_layers: int, activation: str,
                       ffn_dropout: float, bias: bool, norm: nn.Module,
                       drop_path: float = 0., norm_type:str='ln', pre_norm:bool=False, **kwargs):
    encoder_layer = FoxEncoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=d_feedforward,
                                    activation=activation, bias=bias, 
                                    ffn_dropout=ffn_dropout, norm_type=norm_type, pre_norm=pre_norm, **kwargs)


    return FoxEncoder(encoder_layer=encoder_layer, n_layers=n_layers, norm=norm, drop_path=drop_path)