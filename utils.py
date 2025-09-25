from collections import OrderedDict
from os import utime

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


# torch.compile can not track lambda functions. We use nn.Module
class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


class Split(nn.Module):
    def __init__(self, split_size_or_sections, dim):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.split(x, self.split_size_or_sections, self.dim)


class Stack(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.stack(xs, dim=self.dim)


class Cat(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, *xs) -> torch.Tensor:
        return torch.cat(xs, dim=self.dim)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1)


class Function1(nn.Module):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.f(x, *self.args, **self.kwargs)

    def extra_repr(self) -> str:
        return f'{self.f}(x, {self.args}, {self.kwargs})'


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor):
        return a


class Identity2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a, b


class Identity2X(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, *args, **kwargs):
        return a, b


class Identity3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        return a, b, c


class Identity4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d):
        return a, b, c, d


class Identity5(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c, d, e):
        return a, b, c, d, e


class In2Out1st(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a


class In2Out2nd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return b


class Tuple0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tp):
        return tp[0]


class TokenXYTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.stack([x, y], dim=-1)
        z = self.fc(z)
        return z.flatten(start_dim=-2, end_dim=-1)


class OffsetCoordinate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask=None):
        # may be harmful for objection detection tasks
        y -= y[valid_mask].min()
        x -= x[valid_mask].min()
        return p, y, x, valid_mask


class OffsetTime(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor):
        # shape = [B, L]
        return t - t[:, 0].unsqueeze(1)


class FC2(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, activation: str = 'gelu',
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc0 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = create_activation(activation, inplace=True)
        self.fc1 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.act(self.fc0(x))
        return self.fc1(x)


def create_norm(norm: str, d: int, eps: float = 1e-5, bias: bool = True):
    if norm == 'layer':
        norm = nn.LayerNorm(d, eps=eps, bias=bias)
    elif norm == 'rms':
        norm = nn.RMSNorm(d, eps=eps)
    else:
        raise ValueError(norm)

    return norm


def create_activation(act: str, inplace: bool = False):
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
    def __init__(self, d_model: int,
                 dim_feedforward: int,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.activation = create_activation(activation, inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


def create_ffns(
        n_layers: int,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.,
        activation: str = 'relu',
        norm: str = 'layer',
        norm_eps: float = 1e-5,
        bias: bool = True):
    ffns = []
    for i in range(n_layers):
        ffns.append(FFN(d_model, dim_feedforward, dropout, activation, norm, norm_eps, bias))
    return nn.Sequential(*ffns)


def create_transformer_encoder(d_model: int, n_head: int, d_feedforward: int, n_layers: int, dropout: float = 0.,
                               activation: str = 'relu', norm: str = 'layer'):
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
                                               batch_first=True, dropout=dropout, activation=activation)
    norm = create_norm(norm, d_model)
    return nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=norm)


def index_of_tensor_in_list(p: torch.Tensor, p_list: list):
    i = 0
    for item in p_list:
        if p is item:
            return i
        i += 1
    return -1


def configure_param_lr_wd(m: nn.Module, lr: float, wd: float, encoder_lr_decay_rate: float = 0.75,
                          deacy_lr_encoder_layers: nn.Sequential = None, encoder_lr: float = -1):
    p_flag = OrderedDict()
    for p in m.parameters():
        if p.requires_grad:
            key = id(p)
            assert key not in p_flag
            p_flag[key] = {'params': p, 'weight_decay': wd, 'lr': lr}
    weight_decay_modules = m
    if hasattr(m, 'weight_decay_modules'):
        weight_decay_modules = m.weight_decay_modules()

    no_decay_names = ('bias', 'class_token', 'mask_token', 'pos_embedding', 'pe_class_token')

    def if_with_no_decay_names(name):
        for item in no_decay_names:
            if item in name:
                return True
        return False

    for module_name, module in weight_decay_modules.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name

            # check if it needs weight_decay
            if wd > 0 and param.requires_grad:
                if if_with_no_decay_names(param_name) or \
                        isinstance(module, (
                        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.RMSNorm, nn.Embedding,)):
                    p_flag[id(param)]['weight_decay'] = 0.

                elif 'token' in param_name:
                    print(f'{full_param_name} is regarded as with weight decay.')
    if encoder_lr == -1:
        encoder_lr = lr
    if deacy_lr_encoder_layers is not None:
        depth = len(deacy_lr_encoder_layers)
        for i in range(depth):
            dlr = encoder_lr * (encoder_lr_decay_rate ** (depth - i))
            for p in deacy_lr_encoder_layers[i].parameters():
                if p.requires_grad:
                    p_flag[id(p)]['lr'] = dlr

    return p_flag.values()


def norm_tokens(x: torch.Tensor):
    m = x.mean(-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    std = torch.sqrt_(var + 1e-5)
    return (x - m) / std


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def conv_out_length(input_length, kernel_size, stride):
    return torch.floor((input_length - kernel_size) / stride + 1)


class DropSequence(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, tokens: torch.Tensor, valid_mask: torch.BoolTensor):
        if self.training:
            mask = torch.rand(valid_mask.shape, device=valid_mask.device) < self.p
            return tokens, torch.logical_and(mask, valid_mask)
        else:
            return tokens, valid_mask


@torch.compile(fullgraph=True, dynamic=True)
def quantize_timestamp(t: torch.LongTensor, T: int):
    t = t.to(torch.float64)
    # t.shape = [B, L]
    t = t / t.max(1, keepdim=True)[0] * (T - 1)
    t.round_().clamp_(0, T - 1)
    t = t.long()  # quantize t to {0, 1, ..., T-1}
    '''
    reduce the range of T
    '''
    return t


class QuantizeTimestamps(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, t: torch.LongTensor):
        return quantize_timestamp(t, self.T)


def sort_events(x, y, t, p, valid_mask):
    # shape = [B, L]
    dtype = t.dtype
    '''
    经过数据增强后，event本身可能就不连续了，比如会出现 x x pad x x pad x ...
    本函数将event进行重新排序，确保所有的pad在后 
    '''
    t = t.float()
    invalid_mask = ~valid_mask
    t[invalid_mask] = math.inf
    t, index = t.sort(dim=1)
    x = torch.gather(x, dim=1, index=index)
    y = torch.gather(y, dim=1, index=index)
    p = torch.gather(p, dim=1, index=index)
    valid_mask = torch.gather(valid_mask, dim=1, index=index)
    t[~valid_mask] = -1.
    return x, y, t.to(dtype), p, valid_mask


def event_predict_neighbor_cross_entropy(predicts: torch.Tensor, P, H, W, p_, y_, x_, valid_mask_mask, neighbors):
    invalid_mask = ~valid_mask_mask
    assert P == 2
    loss = 0.
    for dx in range(-neighbors, neighbors + 1):
        for dy in range(-neighbors, neighbors + 1):
            x = x_ + dx
            y = y_ + dy
            torch.clamp_(x, 0, W - 1)
            torch.clamp_(y, 0, H - 1)
            targets = p_ * (H * W) + y * W + x
            targets[invalid_mask] = -1
            distance = dx * dx + dy * dy
            loss = loss + F.cross_entropy(predicts.transpose(1, 2), targets, ignore_index=-1) * math.exp(
                - distance / 2.)
    return loss / (2 * neighbors + 1) ** 2


def event_predict_mse(predicts: torch.Tensor, P, H, W, p_, y_, x_, dim_wise_softmax: bool):
    assert P == 2
    # predicts.shape = [B, PHW]
    B = predicts.shape[0]
    if not dim_wise_softmax:
        predicts = F.softmax(predicts, dim=1)
    predicts = predicts.view(B, P, H, W)
    device = predicts.device
    xs = torch.arange(W, device=device, dtype=torch.float).view(1, 1, 1, W)
    ys = torch.arange(H, device=device, dtype=torch.float).view(1, 1, H, 1)
    ps = torch.arange(P, device=device, dtype=torch.float).view(1, P, 1, 1)
    if dim_wise_softmax:
        predicts_x = (F.softmax(predicts.sum(dim=(1, 2), keepdim=True), dim=3) * xs).view(B, -1).sum(1)
        predicts_y = (F.softmax(predicts.sum(dim=(1, 3), keepdim=True), dim=2) * ys).view(B, -1).sum(1)
        predicts_p = (F.softmax(predicts.sum(dim=(2, 3), keepdim=True), dim=1) * ps).view(B, -1).sum(1)
    else:
        predicts_x = (predicts * xs).view(B, -1).sum(1)
        predicts_y = (predicts * ys).view(B, -1).sum(1)
        predicts_p = (predicts * ps).view(B, -1).sum(1)

    predicts = torch.stack((predicts_x, predicts_y, predicts_p))
    targets = torch.stack((x_, y_, p_)).float()
    return F.mse_loss(predicts, targets)


def event_predict_neighbor_accuracy(predicts: torch.Tensor, P, H, W, p_, y_, x_, valid_mask_mask, neighbors):
    assert P == 2
    predicts = predicts.argmax(dim=2)
    corrects = None
    for dx in range(-neighbors, neighbors + 1):
        for dy in range(-neighbors, neighbors + 1):
            x = x_ + dx
            y = y_ + dy
            torch.clamp_(x, 0, W - 1)
            torch.clamp_(y, 0, H - 1)
            targets = p_ * (H * W) + y * W + x
            if corrects is None:
                corrects = predicts == targets
            else:
                corrects = torch.logical_or(corrects, predicts == targets)
    corrects = corrects.float()
    valid_mask_mask = valid_mask_mask.float()
    return (corrects * valid_mask_mask).sum() / (valid_mask_mask.sum() + 1e-5)


def event_predict_neighbor_accuracy_(predicts: torch.Tensor, P, H, W, c, neighbors):
    # c = p * H * W + y * W + x
    x = c % W
    y = (c // W) % H
    p = c // (H * W)
    return event_predict_neighbor_accuracy(predicts=predicts, P=P, H=H, W=W, p_=p, y_=y, x_=x, neighbors=neighbors)


@torch.no_grad()
def calculate_event_prediction_metrics(
        pred_p_logit: torch.Tensor,
        pred_y_norm: torch.Tensor,
        pred_x_norm: torch.Tensor,
        true_p: torch.Tensor,
        true_y: torch.Tensor,
        true_x: torch.Tensor,
        height: int,
        width: int,
        valid_mask: torch.Tensor,
        neighbors: int = 1
) -> dict:
    """
    计算事件预测的各项评估指标。

    Args:
        pred_p_logit (torch.Tensor): 形状为 [B, N] 的模型原始极性 logit 预测。
        pred_y_norm (torch.Tensor): 形状为 [B, N] 的模型归一化 y 坐标预测 (范围在 0-1)。
        pred_x_norm (torch.Tensor): 形状为 [B, N] 的模型归一化 x 坐标预测 (范围在 0-1)。
        true_p (torch.Tensor): 形状为 [B, N] 的真实极性标签 (值为 0 或 1)。
        true_y (torch.Tensor): 形状为 [B, N] 的真实 y 坐标 (像素单位)。
        true_x (torch.Tensor): 形状为 [B, N] 的真实 x 坐标 (像素单位)。
        height (int): 传感器的高度。
        width (int): 传感器的宽度。
        valid_mask (torch.Tensor): 形状为 [B, N] 的布尔或0/1掩码，用于指示哪些是有效样本。
        neighbors (int, optional): 定义邻域的半径。默认为 1，表示一个 3x3 的区域。

    Returns:
        dict: 一个包含多个指标的字典。
    """
    # 确保在无梯度模式下进行计算

    # 1. 预处理预测值
    # 将 logit 转换为类别预测 (0 或 1)
    pred_p = (pred_p_logit > 0).long()

    # 将归一化的坐标反归一化回像素单位，并四舍五入为整数
    pred_y = ((pred_y_norm + 1.) / 2. * (height - 1)).round().long()
    pred_x = ((pred_x_norm + 1.) / 2. * (width - 1)).round().long()

    # 确保预测坐标不会超出图像边界
    pred_y.clamp_(0, height - 1)
    pred_x.clamp_(0, width - 1)

    # 2. 准备掩码和计数
    if valid_mask is not None:
        valid_mask = valid_mask.bool()  # 确保是布尔型
        num_valid = valid_mask.sum()
    else:
        num_valid = pred_p_logit.numel()
    # 如果没有有效样本，返回0
    if num_valid == 0:
        return {
            'p_accuracy': 0.0,
            'mae_y': 0.0,
            'mae_x': 0.0,
            'exact_match_accuracy': 0.0,
            f'neighbor_accuracy': 0.0
        }

    # 3. 计算各项指标

    # 极性准确率
    if valid_mask is not None:
        p_correct = (pred_p[valid_mask] == true_p[valid_mask]).sum()
    else:
        p_correct = (pred_p == true_p).sum()

    p_accuracy = p_correct / num_valid

    # 坐标平均绝对误差 (MAE)
    if valid_mask is not None:
        mae_y = torch.abs(pred_y - true_y).float().mean()
        mae_x = torch.abs(pred_x - true_x).float().mean()
    else:
        mae_y = torch.abs(pred_y[valid_mask] - true_y[valid_mask]).float().mean()
        mae_x = torch.abs(pred_x[valid_mask] - true_x[valid_mask]).float().mean()

    # 精确匹配准确率
    exact_match = (pred_p == true_p) & (pred_y == true_y) & (pred_x == true_x)
    if valid_mask is not None:
        exact_match_correct = exact_match.sum()
    else:
        exact_match_correct = exact_match[valid_mask].sum()
    exact_match_accuracy = exact_match_correct / num_valid

    # 邻居准确率
    p_match = (pred_p == true_p)
    y_is_neighbor = torch.abs(pred_y - true_y) <= neighbors
    x_is_neighbor = torch.abs(pred_x - true_x) <= neighbors

    neighbor_match = p_match & y_is_neighbor & x_is_neighbor
    if valid_mask is not None:
        neighbor_match_correct = neighbor_match[valid_mask].sum()
    else:
        neighbor_match_correct = neighbor_match.sum()

    neighbor_accuracy = neighbor_match_correct / num_valid

    # 4. 准备返回结果
    metrics = {
        'p_accuracy': p_accuracy,
        'mae_y': mae_y,
        'mae_x': mae_x,
        'exact_match_accuracy': exact_match_accuracy,
        f'neighbor_accuracy': neighbor_accuracy
    }

    return metrics


@torch.no_grad()
def calculate_event_prediction_metrics_cls(
        pred_p_logits: torch.Tensor,
        pred_y_logits: torch.Tensor,
        pred_x_logits: torch.Tensor,
        true_p: torch.Tensor,
        true_y: torch.Tensor,
        true_x: torch.Tensor,
        bin_size: int,
        neighbors: int = 1
) -> dict:
    """
    计算事件预测（分类式）的各项评估指标。
    (简化版：假设所有输入都已经是有效的、被mask后的一维张量)

    Args:
        pred_p_logits (torch.Tensor): 形状为 [N, C_p] 的模型原始极性 logit 预测。
        pred_y_logits (torch.Tensor): 形状为 [N, C_y] 的模型 y 坐标 bin 的 logit 预测。
        pred_x_logits (torch.Tensor): 形状为 [N, C_x] 的模型 x 坐标 bin 的 logit 预测。
        true_p (torch.Tensor): 形状为 [N] 的真实极性标签 (值为 0 或 1)。
        true_y (torch.Tensor): 形状为 [N] 的真实 y 坐标 (像素单位)。
        true_x (torch.Tensor): 形状为 [N] 的真实 x 坐标 (像素单位)。
        bin_size (int): 每个坐标轴划分的 bin 的大小（像素）。
        neighbors (int, optional): 定义邻域的半径（像素）。默认为 1。

    Returns:
        dict: 一个包含多个指标的字典。
    """
    num_valid = true_p.numel()
    if num_valid == 0:
        return {
            'p_accuracy': 0.0, 'y_accuracy': 0.0, 'x_accuracy': 0.0,
            'mae_y': 0.0, 'mae_x': 0.0,
            'exact_match_accuracy': 0.0, 'neighbor_accuracy': 0.0
        }

    # 1. 从 Logits 计算预测值
    # 类别预测
    pred_p_class = torch.argmax(pred_p_logits, dim=1)
    pred_y_bin = torch.argmax(pred_y_logits, dim=1)
    pred_x_bin = torch.argmax(pred_x_logits, dim=1)

    # 为了计算 MAE 和邻居准确率，将预测的 bin 转换回像素坐标（取 bin 的中心）
    pred_y_pixel = pred_y_bin * bin_size + bin_size / 2.0
    pred_x_pixel = pred_x_bin * bin_size + bin_size / 2.0

    # 2. 计算真实值的 bin
    true_y_bin = true_y // bin_size
    true_x_bin = true_x // bin_size

    # 3. 计算各项指标
    # 分类准确率
    p_accuracy = (pred_p_class == true_p).float().mean()
    y_accuracy = (pred_y_bin == true_y_bin).float().mean()
    x_accuracy = (pred_x_bin == true_x_bin).float().mean()

    # 坐标平均绝对误差 (MAE)
    mae_y = torch.abs(pred_y_pixel - true_y).float().mean()
    mae_x = torch.abs(pred_x_pixel - true_x).float().mean()

    # 精确匹配准确率 (极性和坐标 bin 都正确)
    exact_match = (pred_p_class == true_p) & (pred_y_bin == true_y_bin) & (pred_x_bin == true_x_bin)
    exact_match_accuracy = exact_match.float().mean()

    # 邻居准确率 (极性正确，且预测的像素坐标在真实坐标的邻域内)
    p_match = (pred_p_class == true_p)
    y_is_neighbor = torch.abs(pred_y_pixel - true_y) <= neighbors
    x_is_neighbor = torch.abs(pred_x_pixel - true_x) <= neighbors
    neighbor_match = p_match & y_is_neighbor & x_is_neighbor
    neighbor_accuracy = neighbor_match.float().mean()

    # 4. 准备返回结果
    metrics = {
        # 分类指标
        'p_accuracy': p_accuracy.item(),
        'y_accuracy': y_accuracy.item(),
        'x_accuracy': x_accuracy.item(),
        # 回归指标 (用于对比)
        'mae_y': mae_y.item(),
        'mae_x': mae_x.item(),
        # 综合指标
        'exact_match_accuracy': exact_match_accuracy.item(),
        'neighbor_accuracy': neighbor_accuracy.item()
    }

    return metrics


def interval_rearrange(v: torch.Tensor, interval: int, reverse: bool):
    assert interval > 1
    v_ = []
    if reverse:
        # [B * interval, L // interval, *] -> [B, L, *]
        dim = v.dim()
        if dim == 2:
            v = v.unsqueeze(2)
        feature_shape = v.shape[2:]

        B = v.shape[0] // interval
        L = v.shape[1] * interval

        v_ = torch.zeros([B, L, v.shape[2]], device=v.device, dtype=v.dtype)
        for i in range(interval):
            v_[:, i::interval] = v[i * B: (i + 1) * B]

        shape_ = [B, L]
        shape_.extend(feature_shape)
        v_ = v_.reshape(shape_)
        if dim == 2:
            v_ = v_.squeeze(2)
        return v_

    else:
        # [B, L, *] -> [B * interval, L // interval, *]
        for i in range(interval):
            v_.append(v[:, i::interval])
        return torch.cat(v_, dim=0)


def sequence_avg_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride: int, avg: bool):
    B, L, d = tokens.shape
    if L % stride != 0:
        print('L % stride != 0')
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


def sequence_cat_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride: int):
    B, L, d = tokens.shape
    assert L % stride == 0
    tokens = (tokens * valid_mask.to(tokens).unsqueeze(2)).view(B, L // stride, stride * d)
    valid_mask = valid_mask.view(B, L // stride, stride).float()
    valid_mask_sum = valid_mask.sum(2)
    valid_mask = valid_mask_sum > 0
    return tokens, valid_mask


def sequence_max_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride: int):
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



def reverse_padded_sequence(x, valid_mask):
    """
    手动反转padding序列中的有效部分
    """
    B, L, d = x.shape
    lengths = valid_mask.sum(dim=1)

    # 1. 创建一个 [0, 1, ..., L-1] 的范围，并扩展到 (B, L)
    arange = torch.arange(L, device=x.device)
    base_indices = arange.unsqueeze(0).expand(B, -1)

    # 2. 计算反转后的索引 (只在有效部分有意义)
    # lengths.unsqueeze(1) -> [B, 1]
    # arange.unsqueeze(0) -> [1, L]
    # 广播后相减
    reversed_indices = lengths.unsqueeze(1) - 1 - base_indices

    # 3. 使用mask决定在每个位置是使用正向索引还是反向索引
    # 在有效部分 (mask=True)，我们使用 reversed_indices
    # 在padding部分 (mask=False)，我们使用 base_indices，以保持padding不动
    # mask 必须扩展到和索引张量一样的形状
    indices = torch.where(valid_mask, reversed_indices, base_indices)

    # 4. 确保索引不会小于0 (对于长度为0的序列)
    indices = indices.clamp(min=0)

    # 5. 扩展索引以匹配x的维度 (B, L) -> (B, L, d)
    indices = indices.unsqueeze(-1).expand(-1, -1, d)

    # 6. 使用gather操作根据新索引采集数据
    x_reversed = torch.gather(x, dim=1, index=indices)
    return x_reversed


import torch


def pad_token_and_valid_mask(tokens: torch.Tensor, valid_mask: torch.Tensor, L: int):
    """
    使用 torch.nn.functional.pad 来高效地填充 token 和 mask。

    Args:
        tokens (torch.Tensor): 输入的 token 张量，形状为 (B, current_L, d)。
        valid_mask (torch.Tensor): 输入的有效位掩码，形状为 (B, current_L)。
        L (int): 目标序列长度。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 填充后的 token 和 valid_mask。
    """
    # 计算需要填充的长度
    padded_len = L - tokens.shape[1]

    # 确保我们是在填充而不是截断
    assert padded_len >= 0, "目标长度 L 必须大于或等于当前 token 长度。"

    # 如果不需要填充，直接返回，避免不必要的操作
    if padded_len == 0:
        return tokens, valid_mask

    # --- 使用 F.pad 进行填充 ---

    # F.pad 的填充参数 `pad` 是一个元组，格式为 (左侧填充数, 右侧填充数, ...)，
    # 从张量的最后一个维度开始指定。

    # 1. 填充 tokens 张量
    # 它的形状是 (B, current_L, d)
    # - 我们不想填充最后一个维度 (d)，所以是 (0, 0)
    # - 我们想在倒数第二个维度 (current_L) 的右侧填充 padded_len 个 0，所以是 (0, padded_len)
    tokens_padded = F.pad(tokens, (0, 0, 0, padded_len), mode='constant', value=0)

    # 2. 填充 valid_mask 张量
    # 它的形状是 (B, current_L)
    # - 我们想在最后一个维度 (current_L) 的右侧填充 padded_len 个 0，所以是 (0, padded_len)
    valid_mask_padded = F.pad(valid_mask, (0, padded_len), mode='constant', value=0)

    return tokens_padded, valid_mask_padded


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    将输入序列的长度减半，同时将特征维度翻倍。

    Args:
        input_dim (int): 输入特征维度。
        group_size (int): 每组用于合并的 token 数量，通常为 2，表示长度减半。
    """

    def __init__(self, input_dim: int, out_dim: int, group_size: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.group_size = group_size
        # LayerNorm有助于稳定训练
        self.norm = nn.LayerNorm(group_size * input_dim)
        # 线性层将拼接后的维度投影到新的维度（通常是2*input_dim）
        self.reduction = nn.Linear(group_size * input_dim, out_dim, bias=False)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor):
        """
        Args:
            tokens (torch.Tensor): 输入序列, shape: (B, L, C).
            valid_mask (torch.Tensor): 有效位掩码, shape: (B, L).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 下采样和升维后的序列, shape: (B, L/group_size, 2*C).
                - 更新后的有效位掩码, shape: (B, L/group_size).
        """
        B, L, C = tokens.shape
        assert C == self.input_dim, f"Input feature dimension ({C}) doesn't match layer's expected dimension ({self.input_dim})."

        # 为了能被 group_size 整除，对序列和掩码进行填充
        padding_needed = (self.group_size - L % self.group_size) % self.group_size
        if padding_needed > 0:
            tokens = F.pad(tokens, (0, 0, 0, padding_needed))
            valid_mask = F.pad(valid_mask, (0, padding_needed), value=False)

        # 更新 L
        L_padded = tokens.shape[1]

        # Reshape & Concatenate
        # (B, L, C) -> (B, L/group_size, group_size, C)
        tokens = tokens.view(B, L_padded // self.group_size, self.group_size, C)
        # (B, L/group_size, group_size, C) -> (B, L/group_size, group_size * C)
        tokens = tokens.flatten(start_dim=2)

        # Projection
        tokens = self.norm(tokens)
        tokens = self.reduction(tokens)

        # 更新 valid_mask
        # (B, L) -> (B, L/group_size, group_size)
        valid_mask = valid_mask.view(B, L_padded // self.group_size, self.group_size)
        # 只要组内有一个token是有效的，我们就认为这个新的聚合token是有效的
        new_valid_mask = valid_mask.any(dim=2)

        return tokens, new_valid_mask


class AttentionPooling(nn.Module):
    """
    一个用于分类任务的注意力池化层。

    该层接收一个序列的隐藏状态 (B, L, d) 和一个可选的掩码 (B, L)，
    然后输出一个池化后的固定维度向量 (B, d)。
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim (int): 输入隐藏状态的维度 (d)。
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. 定义一个可学习的查询向量 (learnable query vector)
        # 这个向量将学会如何“查询”序列中最重要的信息以用于分类。
        # 它的形状是 [d]，但我们用 [1, d] 以方便进行批处理矩阵乘法。
        # 使用 xavier_uniform_ 进行初始化是一种常见的做法。
        self.query_vector = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.query_vector)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): 模型的隐藏状态输出，形状为 [B, L, d]。
            mask (torch.Tensor, optional):
                一个布尔型掩码，形状为 [B, L]。
                值为 True 表示有效 token，False 表示 padding token。
                默认为 None，表示序列中所有 token 都有效。

        Returns:
            torch.Tensor: 池化后的上下文向量，形状为 [B, d]。
        """
        # hidden_states.shape: [B, L, d]
        # query_vector.shape: [1, d]

        # 2. 计算注意力分数 (attention scores)
        # 我们计算查询向量和每个隐藏状态之间的点积相似度。
        # 为了进行批处理，我们将 query_vector 从 [1, d] 扩展到 [B, d]。
        batch_size = hidden_states.size(0)
        query = self.query_vector.expand(batch_size, -1)  # -> [B, d]

        # 使用 bmm (batch matrix multiplication) 计算点积
        # [B, L, d] @ [B, d, 1] -> [B, L, 1]
        # 我们需要将 query 的形状调整为 [B, d, 1]
        scores = torch.bmm(hidden_states, query.unsqueeze(-1)).squeeze(-1)  # -> [B, L]

        # 3. 应用掩码 (apply mask)
        # 如果提供了掩码，我们需要将 padding 位置的分数设为一个非常小的数。
        # 这样在 softmax 后，这些位置的权重会趋近于 0。
        if mask is not None:
            # 使用 `~mask` 来获取 padding 的位置 (False -> True)。
            # 这种写法比 `mask == 0` 更清晰地表达了对布尔张量的操作。
            scores.masked_fill_(~mask, -1e9)

        # 4. 计算注意力权重 (attention weights)
        # 对分数在序列长度维度上进行 softmax，得到归一化的权重。
        # weights.shape: [B, L]
        weights = F.softmax(scores, dim=1)

        # 5. 计算加权和 (weighted sum)
        # 使用权重对隐藏状态进行加权平均。
        # 我们需要将 weights 扩展为 [B, 1, L] 以便与 hidden_states:[B, L, d] 进行 bmm。
        # [B, 1, L] @ [B, L, d] -> [B, 1, d]
        pooled_output = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # -> [B, d]

        return pooled_output



if __name__ == '__main__':
    B = 4
    L = 8
    x = torch.rand([B, L])
    y = torch.rand([B, L])
    t = torch.rand([B, L])
    p = torch.rand([B, L])
    valid_mask = torch.rand([B, L]) < 0.5

    x, y, t, p, valid_mask = sort_events(x, y, t, p, valid_mask)
    print(valid_mask)
    print(t)