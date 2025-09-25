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



class Conv(nn.Module):
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
