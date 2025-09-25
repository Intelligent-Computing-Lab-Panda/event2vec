from modulefinder import Module

import torch
import torch.nn as nn
from math import pi

import torch.nn.functional as F

# also refer to https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
'''
Note that each transform will regard inputs as floats and output floats
So, an additional tranform (ToEventDomain) will be used as the last output to bound coordinates and round them to long
'''


class ToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        return p.float(), y.float(), x.float(), t.float(), valid_mask, target


class ToEventDomain(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W

    def extra_repr(self) -> str:
        return f'H={self.H}, W={self.W}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        y = y.round_()
        x = x.round_()
        p = p.round_()

        mask = (y >= 0) & (y < self.H)
        mask = mask & (x >= 0) & (x < self.W)
        valid_mask = valid_mask & mask

        return p, y, x, t, valid_mask, target


class DiscardTail(nn.Module):
    def __init__(self, scale_y: float, scale_x: float):
        super().__init__()
        self.scale_y = scale_y
        self.scale_x = scale_x

    def extra_repr(self) -> str:
        return f'scale_y={self.scale_y:.6f}, scale_x={self.scale_x:.6f}'

    @staticmethod
    def get_range(v: torch.Tensor, scale: float):
        v_m = v.mean()
        v_std = v.std()
        s = scale * v_std
        return v_m - s, v_m + s

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        # valid_mask.shape = [B, L]

        mask_y = self.get_range(y, self.scale_y)
        mask_x = self.get_range(x, self.scale_x)

        valid_mask = valid_mask & (y >= mask_y[0]) & (y <= mask_y[1])
        valid_mask = valid_mask & (x >= mask_x[0]) & (x <= mask_x[1])

        return p, y, x, t, valid_mask, target


class PositionalNorm(nn.Module):
    def __init__(self, H: int, W: int):
        # it is better to use after DiscardTail to avoid the noise
        super().__init__()
        self.H = H
        self.W = W

    def extra_repr(self) -> str:
        return f'H={self.H}, W={self.W}'

    @staticmethod
    def norm_to_01(v: torch.Tensor):
        v_min = v.min()
        v_max = v.max()

        return (v - v_min) / (v_max - v_min)

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        y = self.norm_to_01(y) * (self.H - 1)
        x = self.norm_to_01(x) * (self.W - 1)

        return p, y, x, t, valid_mask, target


class RadiusFilter(nn.Module):
    def __init__(self, r: float, n_neighbors):
        # only keep events with >= n_neighbors events in neighbor area distance <= r
        super().__init__()
        self.r = r
        assert n_neighbors >= 1
        self.n_neighbors = n_neighbors

    def extra_repr(self) -> str:
        return f'r={self.r:.6f}, n_neighbors={self.n_neighbors}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        # shape = [B, L]
        B, L = x.shape
        dx = x.view(B, L, 1) - x.view(B, 1, L)
        dy = y.view(B, L, 1) - y.view(B, 1, L)

        d = torch.square_(dx) + torch.square_(dy)  # [B, L, L]

        mask = d <= (self.r ** 2)

        mask = mask.long().sum(-1)  # shape = [B, L]
        mask = mask > self.n_neighbors

        return p, y, x, t, valid_mask & mask, target


class RandomErasing(nn.Module):
    def __init__(self, H: int, W: int, prob: float, h: int, w: int):
        super().__init__()
        self.H = H
        self.W = W
        self.prob = prob
        self.h = h
        self.w = w

    def extra_repr(self) -> str:
        return (f'H={self.H}, W={self.W}, prob={self.prob:.6f}, h={self.h}, w={self.w}')

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = p.shape[0]
        cy = torch.rand([B, 1], device=y.device) * (self.H - 1)
        cx = torch.rand([B, 1], device=x.device) * (self.W - 1)

        h = torch.rand([B, 1], device=y.device) * (self.h / 2)
        w = torch.rand([B, 1], device=y.device) * (self.w / 2)

        mask = y >= (cy - h)
        mask = mask & (y <= (cy + h))
        mask = mask & (x >= (cx - w))
        mask = mask & (x <= (cx + w))
        mask_prob = torch.rand([B, 1], device=x.device) < self.prob
        mask = mask & mask_prob
        valid_mask = valid_mask & (~mask)
        return p, y, x, t, valid_mask, target


class Resize(nn.Module):
    def __init__(self, H: int, W: int, h: int, w: int):
        super().__init__()
        self.H = H
        self.W = W
        self.h = h
        self.w = w

    def extra_repr(self) -> str:
        return f'scale_y={self.h / self.H:.6f}, scale_x={self.w / self.W:.6f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        y = (y + 0.5) / self.H * self.h - 0.5
        x = (x + 0.5) / self.W * self.w - 0.5
        return p, y, x, t, valid_mask, target


class RandomResize(nn.Module):
    def __init__(self, scale_y: tuple, scale_x: tuple):
        super().__init__()
        self.scale_y = scale_y
        self.scale_x = scale_x

    def extra_repr(self) -> str:
        return f'scale_y=({self.scale_y[0]:.6f}, {self.scale_y[1]:.6f}), scale_x=({self.scale_x[0]:.6f}, {self.scale_x[1]:.6f})'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = p.shape[0]
        scale_y = torch.empty([B, 1], device=y.device, dtype=torch.float).uniform_(self.scale_y[0], self.scale_y[1])
        scale_x = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(self.scale_x[0], self.scale_x[1])

        y = y * scale_y
        x = x * scale_x
        return p, y, x, t, valid_mask, target


class RandomRotation(nn.Module):
    def __init__(self, H: int, W: int, degrees: float or tuple[float]):
        super().__init__()
        self.H, self.W = H, W
        if isinstance(degrees, tuple):
            assert len(degrees) == 2
            self.degrees = (degrees[0] * pi / 180, degrees[1] * pi / 180)
        elif isinstance(degrees, float):
            self.degrees = (-degrees * pi / 180, degrees * pi / 180)
        else:
            raise ValueError(f'degrees must be either a float or a tuple of floats')
        self.center_y = (H - 1) / 2
        self.center_x = (W - 1) / 2

    def extra_repr(self) -> str:
        return f'center=({self.center_y:.3f}, {self.center_x:.3f}), degrees=({self.degrees[0] * 180 / pi:.3f}, {self.degrees[1] * 180 / pi:.3f})'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        theta = torch.empty([B, 1], device=x.device, dtype=torch.float)
        theta.uniform_(self.degrees[0], self.degrees[1])
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin_(theta)

        x -= self.center_x
        y -= self.center_y

        x_ = x * cos_theta - y * sin_theta
        y_ = x * sin_theta + y * cos_theta

        x_ += self.center_x
        y_ += self.center_y

        return p, y_, x_, t, valid_mask, target


class RandomHorizontalFlip(nn.Module):
    def __init__(self, W: int, prob: float):
        super().__init__()
        self.W = W
        self.prob = prob

    def extra_repr(self) -> str:
        return f'W={self.W}, prob={self.prob:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        mask = torch.rand([B, 1], device=x.device) < self.prob
        return p, y, torch.where(mask, self.W - 1 - x, x), t, valid_mask, target


class RandomVerticalFlip(nn.Module):
    def __init__(self, H: int, prob: float):
        super().__init__()
        self.H = H
        self.prob = prob

    def extra_repr(self) -> str:
        return f'H={self.H}, prob={self.prob:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        mask = torch.rand([B, 1], device=y.device) < self.prob
        return p, torch.where(mask, self.H - 1 - y, y), x, t, valid_mask, target


def crop(y_range: tuple, x_range: tuple, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor,
         valid_mask: torch.BoolTensor):
    mask = (y >= y_range[0]) & (y <= y_range[1])
    mask = mask & (x >= x_range[0]) & (x <= x_range[1])
    # we can return p[mask], y[mask], x[mask], valid_mask[mask]
    # but the number of events (sequence length) will change
    # so, we just set the events to be removed as "padded"
    valid_mask = valid_mask & mask

    return p, y, x, t, valid_mask


class Crop(nn.Module):
    def __init__(self, y_range: tuple, x_range: tuple):
        super().__init__()
        self.y_range, self.x_range = y_range, x_range

    def extra_repr(self) -> str:
        return f'y_range=({self.y_range[0]:.3f}, {self.y_range[1]:.3f}), x_range=({self.x_range[0]:.3f}, {self.x_range[1]:.3f})'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        p, y, x, t, valid_mask = crop(self.y_range, self.x_range, p, y, x, t, valid_mask)
        return p, y, x, t, valid_mask, target


class CenterCrop(Crop):
    def __init__(self, H: int, W: int, size: int or tuple[int]):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            size = (size[0], size[1])
        else:
            raise ValueError(f'size must be either a int or a tuple of ints')
        center_x = W / 2
        center_y = H / 2
        y_range = (center_y - size[0] / 2, center_y + size[0] / 2)
        x_range = (center_x - size[1] / 2, center_x + size[1] / 2)
        super().__init__(y_range, x_range)


class RandomCrop(nn.Module):
    def __init__(self, H: int, W: int, size: int or tuple[int]):
        super().__init__()
        self.H = H
        self.W = W
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.size = (size[0], size[1])
        else:
            raise ValueError(f'size must be either a int or a tuple of ints')

    def extra_repr(self) -> str:
        return f'H={self.H}, W={self.W}, size=({self.size[0]}, {self.size[1]})'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        y_start = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(0, self.H - self.size[0] - 1)
        y_end = y_start + self.size[0]
        x_start = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(0, self.W - self.size[1] - 1)
        x_end = x_start + self.size[1]
        y_range = (y_start, y_end)
        x_range = (x_start, x_end)
        p, y, x, t, valid_mask = crop(y_range, x_range, p, y, x, t, valid_mask)
        return p, y, x, t, valid_mask, target


class RandomCropResize(nn.Module):
    def __init__(self, H: int, W: int, size: int or tuple[int]):
        # resize to the original size after crop
        super().__init__()
        self.H = H
        self.W = W
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            assert len(size) == 2
            self.size = (size[0], size[1])
        else:
            raise ValueError(f'size must be either a int or a tuple of ints')

    def extra_repr(self) -> str:
        return f'H={self.H}, W={self.W}, size=({self.size[0]}, {self.size[1]})'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        y_start = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(0, self.H - self.size[0] - 1)
        y_end = y_start + self.size[0]
        x_start = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(0, self.W - self.size[1] - 1)
        x_end = x_start + self.size[1]
        y_range = (y_start, y_end)
        x_range = (x_start, x_end)
        p, y, x, t, valid_mask = crop(y_range, x_range, p, y, x, t, valid_mask)

        y = (y - y_start) / self.size[0] * (self.H - 1)
        x = (x - x_start) / self.size[1] * (self.W - 1)
        return p, y, x, t, valid_mask, target


class RandomChunkDrop(nn.Module):
    def __init__(self, n_chunk: int, max_mask_len: int):
        super().__init__()
        self.n_chunk = n_chunk
        self.max_mask_len = max_mask_len

    def extra_repr(self) -> str:
        return f'n_chunk={self.n_chunk}, max_mask_len={self.max_mask_len}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B, L = valid_mask.shape
        device = valid_mask.device

        # 1. 生成每个 chunk 的随机长度
        # shape: [B, n_chunk]
        mask_lengths = torch.randint(1, self.max_mask_len + 1, (B, self.n_chunk), device=device)

        mask_lengths = mask_lengths * valid_mask.long().sum(1, keepdim=True) / L
        # 按实际长度比例进行缩放，避免事件数量太少的样本被全部mask

        # 2. 生成每个 chunk 的随机起始位置
        # 为防止区间超出边界，我们从 [0, L - 1] 中采样
        # shape: [B, n_chunk]
        mask_starts = torch.randint(0, L, (B, self.n_chunk), device=device)

        # 3. 构建掩码 (高效的向量化方法)
        # 创建一个序列索引 [0, 1, ..., L-1]
        indices = torch.arange(L, device=device).unsqueeze(0).unsqueeze(1)  # shape: [1, 1, L]

        # 将 starts 和 lengths 扩展维度以进行广播
        starts = mask_starts.unsqueeze(2)  # shape: [B, n_chunk, 1]
        lengths = mask_lengths.unsqueeze(2)  # shape: [B, n_chunk, 1]
        ends = starts + lengths  # shape: [B, n_chunk, 1]

        # 利用广播机制判断每个位置是否在任何一个 chunk 内
        # (indices >= starts) & (indices < ends) 会生成一个 [B, n_chunk, L] 的布尔张量
        # .any(dim=1) 在 n_chunk 维度上做 OR 操作，得到最终的掩码
        mask = (indices >= starts) & (indices < ends)
        mask = mask.any(dim=1)  # shape: [B, L]

        # 4. 应用掩码
        valid_mask[mask] = False

        return p, y, x, t, valid_mask, target


class RandomShear(nn.Module):
    def __init__(self, sy: float, sx: float):
        super().__init__()
        self.sy = sy
        self.sx = sx

    def extra_repr(self) -> str:
        return f'sy={self.sy:.3f}, sx={self.sx:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        sy = torch.empty([B, 1], device=y.device, dtype=torch.float).uniform_(-self.sy, self.sy)
        sx = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(-self.sx, self.sx)

        y_ = y + sy * x
        x_ = x + sx * y

        return p, y_, x_, t, valid_mask, target


class RandomTranslate(nn.Module):
    def __init__(self, dy: int, dx: int):
        super().__init__()
        self.dy = dy
        self.dx = dx

    def extra_repr(self) -> str:
        return f'dy={self.dy}, dx={self.dx}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        B = x.shape[0]
        dy = torch.empty([B, 1], device=y.device, dtype=torch.float).uniform_(-self.dy, self.dy)
        dx = torch.empty([B, 1], device=x.device, dtype=torch.float).uniform_(-self.dx, self.dx)
        y = y + dy
        x = x + dx

        return p, y, x, t, valid_mask, target


class RandomTemporalWrap(nn.Module):
    def __init__(self, scale_low: float, scale_high: float):
        super().__init__()
        self.scale_low = scale_low
        self.scale_high = scale_high
        assert scale_high > scale_low

    def extra_repr(self) -> str:
        return f'scale={self.scale_low:.3f}, {self.scale_high:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        if t.dtype != torch.float:
            print(t.dtype)
            exit(-1)
        '''
        考虑到valid_mask，t可能是类似
        0 1 2 3 4 x x x ...
        diff得到
        1 1 1 1 x x x x ..
        cumsum得到
        1 2 3 4

        '''
        invalid_mask = ~valid_mask
        diff_t = torch.diff(t, dim=1)
        diff_t[invalid_mask[:, 1:]] = 0.
        scale = torch.rand_like(diff_t) * (self.scale_high - self.scale_low) + self.scale_low
        diff_t *= scale
        t = torch.cat((t[:, 0:1], t[:, 0:1] + torch.cumsum(diff_t, dim=1)), dim=1)
        t[invalid_mask] = -1.
        return p, y, x, t, valid_mask, target


class RandomChunkWrap(nn.Module):
    def __init__(self, n_chunk: int, max_mask_len: int, scale_low: float, scale_high: float):
        super().__init__()
        self.n_chunk = n_chunk
        self.max_mask_len = max_mask_len
        self.scale_low = scale_low
        self.scale_high = scale_high

    def extra_repr(self) -> str:
        return f'n_chunk={self.n_chunk}, max_mask_len={self.max_mask_len}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        if t.dtype != torch.float:
            print(t.dtype)
            exit(-1)
        B, L = valid_mask.shape
        device = valid_mask.device

        # 1. 生成每个 chunk 的随机长度
        # shape: [B, n_chunk]
        mask_lengths = torch.randint(1, self.max_mask_len + 1, (B, self.n_chunk), device=device)

        # 2. 生成每个 chunk 的随机起始位置
        # 为防止区间超出边界，我们从 [0, L - 1] 中采样
        # shape: [B, n_chunk]
        mask_starts = torch.randint(0, L, (B, self.n_chunk), device=device)

        # 3. 构建掩码 (高效的向量化方法)
        # 创建一个序列索引 [0, 1, ..., L-1]
        indices = torch.arange(L, device=device).unsqueeze(0).unsqueeze(1)  # shape: [1, 1, L]

        # 将 starts 和 lengths 扩展维度以进行广播
        starts = mask_starts.unsqueeze(2)  # shape: [B, n_chunk, 1]
        lengths = mask_lengths.unsqueeze(2)  # shape: [B, n_chunk, 1]
        ends = starts + lengths  # shape: [B, n_chunk, 1]

        # 利用广播机制判断每个位置是否在任何一个 chunk 内
        # (indices >= starts) & (indices < ends) 会生成一个 [B, n_chunk, L] 的布尔张量
        # .any(dim=1) 在 n_chunk 维度上做 OR 操作，得到最终的掩码
        mask = (indices >= starts) & (indices < ends)
        mask = mask.any(dim=1)  # shape: [B, L]

        # 4. 应用掩码
        mask = valid_mask & mask
        t_m = t[mask]
        scale = torch.rand_like(t_m) * (self.scale_high - self.scale_low) + self.scale_low
        t[mask] = t_m * scale
        return p, y, x, t, valid_mask, target


class RandomReplaceByNoise(nn.Module):
    def __init__(self, H: int, W: int, p_replace: float):
        super().__init__()
        self.H = H
        self.W = W
        self.p_replace = p_replace

    def extra_repr(self) -> str:
        return f'H={self.H}, W={self.W}, p_replace={self.p_replace:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        replace_mask = torch.rand_like(p, dtype=torch.float) < self.p_replace
        replace_mask = replace_mask.logical_and(valid_mask)

        noise_x = torch.rand_like(x, dtype=torch.float) * (self.W - 1)
        noise_y = torch.rand_like(y, dtype=torch.float) * (self.H - 1)
        noise_p = torch.randint(low=0, high=2, size=p.shape, device=p.device).float()

        x[replace_mask] = noise_x[replace_mask]
        y[replace_mask] = noise_y[replace_mask]
        p[replace_mask] = noise_p[replace_mask]
        return p, y, x, t, valid_mask, target


class RandomDrop(nn.Module):
    def __init__(self, p_drop: float):
        super().__init__()
        self.p_drop = p_drop

    def extra_repr(self) -> str:
        return f'p_drop={self.p_drop}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        keep_mask = torch.rand_like(p, dtype=torch.float) > self.p_drop
        valid_mask = torch.logical_and(valid_mask, keep_mask)
        return p, y, x, t, valid_mask, target


class RandomPolarityFlip(nn.Module):
    def __init__(self, p_flip: float):
        super().__init__()
        self.p_flip = p_flip

    def extra_repr(self) -> str:
        return f'p_flip={self.p_flip}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        mask = torch.rand_like(p) < self.p_flip
        mask = mask & valid_mask
        p[mask] = 1 - p[mask]
        return p, y, x, t, valid_mask, target



class Sequential(nn.Sequential):
    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        for module in self:
            p, y, x, t, valid_mask, target = module(p, y, x, t, valid_mask, target)
        return p, y, x, t, valid_mask, target



class RandomApply(nn.ModuleList):
    def __init__(self, prob: float, *args):
        super().__init__()
        self.prob = prob
        for m in args:
            self.append(m)

    def extra_repr(self) -> str:
        return super().extra_repr() + f', prob={self.prob:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        mask = torch.rand(len(self)) < self.prob
        for i in range(len(self)):
            if mask[i]:
                p, y, x, t, valid_mask, target = self[i](p, y, x, t, valid_mask, target)

        return p, y, x, t, valid_mask, target


class RandomShuffleApply(nn.ModuleList):
    def __init__(self, prob: float, *args):
        super().__init__()
        self.prob = prob
        for m in args:
            self.append(m)

    def extra_repr(self) -> str:
        return super().extra_repr() + f', prob={self.prob:.3f}'

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        mask = torch.rand(len(self)) < self.prob
        indices = torch.randperm(len(self))
        for i in range(len(self)):
            if mask[i]:
                p, y, x, t, valid_mask, target = self[indices[i].item()](p, y, x, t, valid_mask, target)

        return p, y, x, t, valid_mask, target


class RandomChoice(nn.ModuleList):
    def __init__(self, *args):
        super().__init__()
        for arg in args:
            self.append(arg)

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        i = torch.randint(low=0, high=len(self), size=[1]).item()
        return self[i](p, y, x, t, valid_mask, target)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, t: torch.Tensor, valid_mask: torch.BoolTensor, target: torch.Tensor):
        return p, y, x, t, valid_mask, target


class TransformWrapper(nn.Module):
    def __init__(self, train_transforms: nn.Module, test_transforms: nn.Module):
        super().__init__()
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def forward(self, p, y, x, t, valid_mask, target):
        if self.training:
            return self.train_transforms(p, y, x, t, valid_mask, target)
        else:
            return self.test_transforms(p, y, x, t, valid_mask, target)




def parse_transforms(transform_args: str):
    transform_args = transform_args.split('/')
    transforms = []

    for item in transform_args:
        if item.startswith('DiscardTail-'):
            # DiscardTail-{scale}
            item = item.split('-')[1].split(',')
            scale_y = float(item[0])
            scale_x = float(item[1])
            transforms.append(DiscardTail(scale_y, scale_x))

        elif item.startswith('PositionalNorm-'):
            # PositionalNorm-{H,W}
            item = item.split('-')[1].split(',')
            H = int(item[0])
            W = int(item[1])
            transforms.append(PositionalNorm(H, W))

        elif item.startswith('RadiusFilter-'):
            # RadiusFilter-{r,n_neighbors}
            item = item.split('-')[1].split(',')
            r = float(item[0])
            n_neighbors = int(item[1])
            transforms.append(RadiusFilter(r, n_neighbors))

        elif item.startswith('RandomErasing-'):
            # RandomErasing-{H,W}-{prob}-{h,w}
            item = item.split('-')
            HW = item[1].split(',')
            H = int(HW[0])
            W = int(HW[1])
            prob = float(item[2])
            hw = item[3].split(',')
            h = int(hw[0])
            w = int(hw[1])
            transforms.append(RandomErasing(H=H, W=W, prob=prob, h=h, w=w))


        elif item.startswith('Resize-'):
            # Resize-{H,W}-{h,w}
            item = item.split('-')
            temp = item[1].split(',')
            H = int(temp[0])
            W = int(temp[1])
            temp = item[2].split(',')
            h = int(temp[0])
            w = int(temp[1])
            transforms.append(Resize(H=H, W=W, h=h, w=w))

        elif item.startswith('RandomResize-'):
            # RandomResize-{scale_y[0],scale_y[1]}-{scale_x[0],scale_x[1]}
            item = item.split('-')
            scale_y = item[1].split(',')
            scale_x = item[2].split(',')
            scale_y = (float(scale_y[0]), float(scale_y[1]))
            scale_x = (float(scale_x[0]), float(scale_x[1]))
            transforms.append(RandomResize(scale_y, scale_x))

        elif item.startswith('RandomRotation-'):
            # RandomRotation-{H,W}-{degrees}
            # RandomRotation-{H,W}-{degrees[0],degrees[1]}
            item = item.split('-')
            HW = item[1].split(',')
            H = int(HW[0])
            W = int(HW[1])
            if ',' in item[2]:
                degrees = item[2].split(',')
                degrees = (float(degrees[0]), float(degrees[1]))
            else:
                degrees = float(item[2])
            transforms.append(RandomRotation(H, W, degrees))

        elif item.startswith('RandomHorizontalFlip-'):
            # RandomHorizontalFlip-{W}-{prob}
            item = item.split('-')
            W = int(item[1])
            prob = float(item[2])
            transforms.append(RandomHorizontalFlip(W, prob))

        elif item.startswith('RandomVerticalFlip-'):
            # RandomVerticalFlip-{H}-{prob}
            item = item.split('-')
            H = int(item[1])
            prob = float(item[2])
            transforms.append(RandomVerticalFlip(H, prob))

        elif item.startswith('Crop-'):
            # Crop-{y_range[0],y_range[1]}-{x_range[0],x_range[1]}
            item = item.split('-')
            y_range = item[1].split(',')
            y_range = (int(y_range[0]), int(y_range[1]))
            x_range = item[2].split(',')
            x_range = (int(x_range[0]), int(x_range[1]))
            transforms.append(Crop(y_range, x_range))

        elif item.startswith('CenterCrop-'):
            # CenterCrop-{H,W}-{size}
            # CenterCrop-{H,W}-{size[0],size[1]}
            item = item.split('-')
            HW = item[1].split(',')
            H = int(HW[0])
            W = int(HW[1])

            if ',' in item[2]:
                size = item[2].split(',')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(item[2])

            transforms.append(CenterCrop(H, W, size))

        elif item.startswith('RandomCrop-'):
            # RandomCrop-{H,W}-{size}
            # RandomCrop-{H,W}-{size[0],size[1]}
            item = item.split('-')
            HW = item[1].split(',')
            H = int(HW[0])
            W = int(HW[1])

            if ',' in item[2]:
                size = item[2].split(',')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(item[2])

            transforms.append(RandomCrop(H, W, size))

        elif item.startswith('RandomCropResize-'):
            # RandomCropResize-{H,W}-{size}
            # RandomCropResize-{H,W}-{size[0],size[1]}
            item = item.split('-')
            HW = item[1].split(',')
            H = int(HW[0])
            W = int(HW[1])

            if ',' in item[2]:
                size = item[2].split(',')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(item[2])

            transforms.append(RandomCropResize(H, W, size))

        elif item.startswith('RandomShear-'):
            # RandomShear-{sy,sx}
            item = item.split('-')[1].split(',')
            sy = float(item[0])
            sx = float(item[1])
            transforms.append(RandomShear(sy, sx))

        elif item.startswith('RandomTranslate-'):
            # RandomTranslate-{dy,dx}
            item = item.split('-')[1].split(',')
            dy = int(item[0])
            dx = int(item[1])
            transforms.append(RandomTranslate(dy, dx))

        elif item.startswith('RandomChunkDrop-'):
            # RandomChunkDrop-{n_chunk,max_mask_len}
            item = item.split('-')[1].split(',')
            n_chunk = int(item[0])
            max_mask_len = int(item[1])
            transforms.append(RandomChunkDrop(n_chunk, max_mask_len))

        elif item.startswith('RandomChunkWrap-'):
            # RandomChunkWrap-{n_chunk,max_mask_len}-{scale_low,scale_high}
            item = item.split('-')
            temp = item[1].split(',')
            n_chunk = int(temp[0])
            max_mask_len = int(temp[1])

            temp = item[2].split(',')
            scale_low = float(temp[0])
            scale_high = float(temp[1])
            transforms.append(RandomChunkWrap(n_chunk, max_mask_len, scale_low, scale_high))


        elif item.startswith('RandomPolarityFlip-'):
            # RandomPolarityFlip-{p_flip}
            p_flip = float(item.split('-')[1])
            transforms.append(RandomPolarityFlip(p_flip))

        elif item.startswith('RandomTemporalWrap-'):
            # RandomTemporalWrap-{scale_low,scale_high}
            item = item.split('-')[1].split(',')
            scale_low = float(item[0])
            scale_high = float(item[1])
            transforms.append(RandomTemporalWrap(scale_low, scale_high))

        elif item.startswith('RandomDrop-'):
            # RandomDrop-{p_drop}
            p_drop = float(item.split('-')[1])
            transforms.append(RandomDrop(p_drop))

        elif item.startswith('RandomReplaceByNoise-'):
            # RandomReplaceByNoise-{H,W}-{p_replace}
            item = item.split('-')
            temp = item[1].split(',')
            H = int(temp[0])
            W = int(temp[1])
            p_replace = float(item[2])
            transforms.append(RandomReplaceByNoise(H=H, W=W, p_replace=p_replace))

        else:
            raise NotImplementedError(item)

    return transforms


# class EventTransforms(Sequential):
#     def forward(self, p: torch.Tensor, y: torch.Tensor, x: torch.Tensor, valid_mask: torch.BoolTensor):
#         # 使用分组采样时，数据就需要展平了
#         shape = p.shape
#         p = p.flatten(1)
#         y = y.flatten(1)
#         x = x.flatten(1)
#         valid_mask = valid_mask.flatten(1)
#         p, y, x, valid_mask = super().forward(p, y, x, valid_mask)
#         p = p.reshape(shape)
#         y = y.reshape(shape)
#         x = x.reshape(shape)
#         valid_mask = valid_mask.reshape(shape)
#         return p, y, x, valid_mask


def get_transform_module(train_transform_args: str, train_transform_policy: str, test_transform_args: str, H: int,
                         W: int, h: int, w: int):
    if train_transform_args is not None and train_transform_args != 'none' and train_transform_args != '':
        train_transforms = parse_transforms(train_transform_args)
        if train_transform_policy == 'Sequential':
            train_transforms = Sequential(*train_transforms)
        elif train_transform_policy.startswith('RandomApply-'):
            # RandomApply-{prob}
            prob = float(train_transform_policy.split('-')[1])
            train_transforms = RandomApply(prob, *train_transforms)

        elif train_transform_policy.startswith('RandomShuffleApply-'):
            # RandomShuffleApply-{prob}
            prob = float(train_transform_policy.split('-')[1])
            train_transforms = RandomShuffleApply(prob, *train_transforms)

        elif train_transform_policy == 'RandomChoice':
            train_transforms.append(Identity())
            train_transforms = RandomChoice(*train_transforms)

        else:
            raise ValueError(train_transform_policy)

    else:
        train_transforms = Identity()

    if test_transform_args is None or test_transform_args == '' or test_transform_args == 'none':
        test_transforms = Identity()
    else:

        test_transforms = Sequential(*parse_transforms(test_transform_args))

    tfs = [
        ToFloat(),
        TransformWrapper(train_transforms=train_transforms, test_transforms=test_transforms),
    ]

    if h == H and w == W:
        tfs.append(ToEventDomain(H, W))
        pass
    else:
        tfs.append(Resize(H=H, W=W, h=h, w=w))
        tfs.append(ToEventDomain(h, w))

    tfs = Sequential(*tfs)
    # tfs = torch.compile(tfs, dynamic=False)
    return tfs


# token-level transform

class TokenMix(nn.Module):
    """
    Applies TokenMix augmentation to a batch of token embeddings.
    This module should be placed after the event embedding layer and
    before the main Transformer encoder blocks.

    Reference: "TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers"
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 100):
        """
        Args:
            alpha (float): The hyperparameter for the Beta distribution,
                           controlling the mixing ratio.
            num_classes (int): The total number of classes for one-hot encoding targets.
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        # Initialize Beta distribution if alpha > 0 for mixing
        if self.alpha > 0:
            self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        print(f"TokenMix initialized with alpha={alpha}, num_classes={num_classes}")

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            tokens (torch.Tensor): The input token embeddings,
                                   shape should be [B, L, D]
                                   B: batch size, L: sequence length, D: embedding dimension.
            targets (torch.Tensor): The ground truth labels, shape should be [B].

        Returns:
            torch.Tensor: The mixed token embeddings.
            torch.Tensor: The mixed one-hot encoded targets.
        """
        # Ensure we are in training mode to apply augmentation
        if not self.training or self.alpha <= 0:
            # If not training or alpha is 0, return original inputs
            # but ensure targets are one-hot encoded for consistency
            return tokens, F.one_hot(targets, num_classes=self.num_classes).float()

        B, L, D = tokens.shape
        device = tokens.device

        # 1. Create a shuffled batch to mix with
        shuffled_indices = torch.randperm(B, device=device)
        tokens_b = tokens[shuffled_indices]
        targets_b = targets[shuffled_indices]

        # 2. Sample the mixing ratio lambda
        # This lambda represents the proportion of tokens to keep from the original batch
        lam = self.beta_distribution.sample().to(device)

        # 3. Determine the number of tokens to replace from batch_b
        # Note: The original paper mixes (1-lam) ratio. We follow that.
        # Number of tokens to take from batch_b
        num_to_mix = int((1 - lam) * L)

        if num_to_mix == 0:
            # No mixing needed, return original
            return tokens, F.one_hot(targets, num_classes=self.num_classes).float()

        # 4. Select random indices of tokens to be replaced
        mix_indices = torch.randperm(L, device=device)[:num_to_mix]

        # 5. Perform the token mixing
        tokens_mixed = tokens.clone()
        # Replace the tokens at the selected indices with tokens from the shuffled batch
        tokens_mixed[:, mix_indices, :] = tokens_b[:, mix_indices, :]

        # 6. Adjust lambda based on the actual number of mixed tokens
        # This provides a more precise label mixing ratio
        true_lam = 1.0 - (len(mix_indices) / L)

        # 7. Mix the targets
        targets_a_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_b_one_hot = F.one_hot(targets_b, num_classes=self.num_classes).float()

        mixed_targets = true_lam * targets_a_one_hot + (1.0 - true_lam) * targets_b_one_hot

        return tokens_mixed, mixed_targets

class RandomMixup(nn.Module):
    """
    对输入的批次和目标随机应用 Mixup。
    该类实现了论文 "mixup: Beyond Empirical Risk Minimization" 中描述的数据增强方法。
    此版本已适配 (B, L, D) 形状的三维 Token 嵌入张量。

    Args:
        num_classes (int): 用于 one-hot 编码的类别总数。
        p (float): 对批次进行数据增强的概率。默认值为 0.5。
        alpha (float): 用于 Mixup 的 Beta 分布的超参数。默认值为 1.0。
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须是正整数。")
        if alpha <= 0:
            raise ValueError("alpha 参数不能为零或负数。")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        # 创建 Beta 分布采样器
        self.beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): 形状为 (B, L, D) 的浮点张量。
            target (Tensor): 形状为 (B,) 的整数张量。

        Returns:
            Tensor: 经过随机 Mixup 变换后的批次。
            Tensor: 混合后的目标。
        """
        # --- 输入验证 ---
        if batch.ndim != 3:
            raise ValueError(f"批次张量的维度应为 3 (B, L, D)，但得到 {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"目标张量的维度应为 1，但得到 {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"批次张量的数据类型应为浮点型，但得到 {batch.dtype}。")
        if target.dtype != torch.int64:
            raise TypeError(f"目标张量的数据类型应为 torch.int64，但得到 {target.dtype}")

        # --- 根据概率 p 应用数据增强 ---
        if torch.rand(1).item() >= self.p:
            return batch, target

        # --- 将目标进行 one-hot 编码 ---
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # --- 创建用于混合的批次对 ---
        # 通过滚动（roll）操作创建配对，比随机打乱（shuffle）更高效
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # --- 生成混合比例 (lambda) ---
        lambda_param = self.beta.sample().to(batch.device)

        # --- 应用 Mixup ---
        # 广播机制会自动处理 (B, L, D) 的情况
        batch_mixed = batch * lambda_param + batch_rolled * (1.0 - lambda_param)
        target_mixed = target * lambda_param + target_rolled * (1.0 - lambda_param)

        return batch_mixed, target_mixed


class RandomCutmix(nn.Module):
    """
    对输入的批次和目标随机应用 Cutmix。
    该类实现了论文 "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" 中描述的数据增强方法。
    此版本已适配 (B, L, D) 形状的三维 Token 嵌入张量。

    Args:
        num_classes (int): 用于 one-hot 编码的类别总数。
        p (float): 对批次进行数据增强的概率。默认值为 0.5。
        alpha (float): 用于 Cutmix 的 Beta 分布的超参数。默认值为 1.0。
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须是正整数。")
        if alpha <= 0:
            raise ValueError("alpha 参数不能为零或负数。")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        # 创建 Beta 分布采样器
        self.beta = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha))

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): 形状为 (B, L, D) 的浮点张量。
            target (Tensor): 形状为 (B,) 的整数张量。

        Returns:
            Tensor: 经过随机 Cutmix 变换后的批次。
            Tensor: 混合后的目标。
        """
        # --- 输入验证 ---
        if batch.ndim != 3:
            raise ValueError(f"批次张量的维度应为 3 (B, L, D)，但得到 {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"目标张量的维度应为 1，但得到 {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"批次张量的数据类型应为浮点型，但得到 {batch.dtype}。")
        if target.dtype != torch.int64:
            raise TypeError(f"目标张量的数据类型应为 torch.int64，但得到 {target.dtype}")

        # --- 根据概率 p 应用数据增强 ---
        if torch.rand(1).item() >= self.p:
            return batch, target

        # --- 将目标进行 one-hot 编码 ---
        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # --- 创建用于混合的批次对 ---
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # --- 生成混合比例 (lambda) ---
        lambda_param = self.beta.sample().to(batch.device)

        # --- 生成剪切区域的坐标 ---
        B, L, D = batch.shape
        cut_ratio = 1. - lambda_param
        cut_w = (L * cut_ratio).long()

        cx = torch.randint(0, L, (1,), device=batch.device)[0]
        bbx1 = torch.clamp(cx - cut_w // 2, 0, L)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, L)

        # --- 创建掩码并应用剪切区域 ---
        # 创建一个 (B, L, 1) 的掩码，以便它可以广播到 (B, L, D)
        mask = torch.ones(B, L, 1, device=batch.device, dtype=batch.dtype)
        mask[:, bbx1:bbx2, :] = 0  # 在剪切区域将掩码设置为 0

        batch_mixed = batch * mask + batch_rolled * (1. - mask)

        # --- 根据实际剪切区域大小调整 lambda 并混合标签 ---
        actual_lambda = 1 - ((bbx2 - bbx1).float() / L)
        target_mixed = target * actual_lambda + target_rolled * (1. - actual_lambda)

        return batch_mixed, target_mixed


class TokenChunkMasking(nn.Module):
    """
    一个数据增强模块，对输入的 token 序列进行随机块掩码 (置零)。

    该模块会随机选择 n_chunk 个起始位置，并从每个位置开始，
    将长度为 1 到 max_mask_len 之间的随机长度的连续 token 置零。
    这种方法常用于各种自监督学习任务中。
    """

    def __init__(self, n_chunk: int, max_mask_len: int):
        """
        初始化函数。

        参数:
            n_chunk (int): 需要置零的块（chunk）的数量。
            max_mask_len (int): 每个块（chunk）的最大掩码长度。
                                实际掩码长度会从 [1, max_mask_len] 中随机选择。
        """
        super().__init__()
        if not isinstance(n_chunk, int) or n_chunk <= 0:
            raise ValueError("n_chunk 必须是一个正整数。")
        if not isinstance(max_mask_len, int) or max_mask_len <= 0:
            raise ValueError("max_mask_len 必须是一个正整数。")

        self.n_chunk = n_chunk
        self.max_mask_len = max_mask_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, L, d]，其中
                              B 是批量大小 (batch size),
                              L 是序列长度 (sequence length),
                              d 是特征维度 (feature dimension)。

        返回:
            torch.Tensor: 经过随机块掩码处理后的张量，形状与输入相同。
        """
        # 获取输入张量的维度信息
        B, L, d = x.shape
        device = x.device

        # --- 步骤 1: 为每个批次中的每个 chunk 生成随机的起始位置 ---
        # start_indices 的形状为 [B, n_chunk]
        # randint 的上界是 L，表示生成的索引范围是 [0, L-1]
        start_indices = torch.randint(0, L, (B, self.n_chunk), device=device)

        # --- 步骤 2: 为每个 chunk 生成随机的掩码长度 ---
        # mask_lengths 的形状为 [B, n_chunk]
        # 长度范围从 1 到 max_mask_len (包含)
        mask_lengths = torch.randint(1, self.max_mask_len + 1, (B, self.n_chunk), device=device)

        # --- 步骤 3: 使用广播机制高效地创建掩码 ---
        # 创建一个序列索引张量，形状为 [L]，并调整为 [1, 1, L] 以便广播
        indices = torch.arange(L, device=device).view(1, 1, L)

        # 调整 start_indices 和 mask_lengths 的形状为 [B, n_chunk, 1] 以便广播
        start = start_indices.unsqueeze(2)
        # 计算每个 chunk 的结束位置（不包含）
        end = (start_indices + mask_lengths).unsqueeze(2)

        # 创建一个布尔类型的掩码。
        # 当一个 token 的索引 `j` 满足 `start_idx <= j < end_idx` 时，它将被掩码。
        # 通过广播，`condition` 的形状将变为 [B, n_chunk, L]
        condition = (indices >= start) & (indices < end)

        # 沿着 n_chunk 维度进行 "或" 操作。
        # 只要任何一个 chunk 覆盖了某个 token，该 token 就应该被掩码。
        # final_mask 的形状为 [B, L]
        final_mask = torch.any(condition, dim=1)

        # --- 步骤 4: 应用掩码 ---
        # 复制输入张量，以避免在原始输入上进行原地修改
        x_masked = x.clone()

        # 将 final_mask 扩展为 [B, L, 1] 以匹配 x_masked 的形状
        # 然后使用 masked_fill_ 将掩码位置的元素置零
        x_masked.masked_fill_(final_mask.unsqueeze(-1), 0.)

        return x_masked


if __name__ == '__main__':

    from PIL import Image
    from matplotlib import pyplot as plt
    from torchvision import transforms

    H = 128
    W = 128
    image = Image.open('./transform_sample.bmp')
    to_tensor = transforms.PILToTensor()
    image = to_tensor(image)
    assert isinstance(image, torch.Tensor), type(image)
    image = image.reshape(H, W)
    image_ = image.clone()
    image = image.flatten()
    indices = torch.arange(H * W)[~image]
    # indices = y * W + x
    y = indices // W
    x = indices - y * W
    p = torch.zeros_like(y, dtype=torch.bool)
    valid_mask = torch.ones_like(y, dtype=torch.bool)
    h = H
    w = W
    tf = Sequential(
        ToFloat(),
        # RandomResize((0.3, 0.7), (0.3, 0.7)),
        RandomRotation(h, w, 15.),
        RandomHorizontalFlip(w, 0.5),
        RandomVerticalFlip(h, 0.5),
        # RandomCrop(h, w, 64),
        RandomShear(0.2, 0.2),
        RandomTranslate(16, 16),
        ToEventDomain(h, w)
    )
    p_ = p.clone()
    y_ = y.clone()
    x_ = x.clone()
    valid_mask_ = valid_mask.clone()
    images = []
    for _ in range(5):
        p, y, x, valid_mask = tf(p_.unsqueeze(0), y_.unsqueeze(0), x_.unsqueeze(0), valid_mask_.unsqueeze(0))
        p = p.squeeze(0)
        y = y.squeeze(0)
        x = x.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

        y = y[valid_mask]
        x = x[valid_mask]

        indices = y * w + x
        image = torch.ones(h * w)
        image[indices] = 0
        image = image.reshape(h, w)
        images.append(image)
    n = len(images)
    plt.subplot(1, n, 1)
    plt.imshow(image_.numpy(), cmap='gray')

    for i in range(1, n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].numpy(), cmap='gray')

    plt.show()




