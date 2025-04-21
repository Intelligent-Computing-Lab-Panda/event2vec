import datetime
import math
import time
from typing import Callable, Any
from collections import OrderedDict


import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import Accuracy


class OffsetCoordinate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask = None):
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

class Neighbor1D(nn.Module):
    def __init__(self, L:int, n: int, dilated:int):
        # create 2n+1 neighbors of x
        super().__init__()
        self.n = n
        self.L = L
        self.dilated = dilated

    def forward(self, x: torch.LongTensor):
        # x.shape = [B, T]
        dx = (torch.arange(-self.n, self.n + 1, device=x.device, dtype=torch.long) * self.dilated).view(-1, 1, 1)
        x = x.unsqueeze(0) + dx  # [2n + 1, B, T]
        torch.clamp_(x, min=0, max=self.L - 1)
        return x

class Neighbor2D(nn.Module):
    def __init__(self, H:int, W:int, n: int, dilated:int):
        super().__init__()
        self.H = H
        self.W = W
        self.n = n
        self.dilated = dilated

    def forward(self, y: torch.LongTensor, x: torch.LongTensor):
        dx = (torch.arange(-self.n, self.n + 1, device=x.device, dtype=torch.long) * self.dilated).view(-1, 1, 1)
        x = x.unsqueeze(0) + dx  # [2n + 1, B, T]
        torch.clamp_(x, min=0, max=self.W - 1)

        dy = (torch.arange(-self.n, self.n + 1, device=y.device, dtype=torch.long) * self.dilated).view(-1, 1, 1)
        y = y.unsqueeze(0) + dy  # [2n + 1, B, T]
        torch.clamp_(y, min=0, max=self.H - 1)

        return (y.unsqueeze(1) + x).flatten(0, 1)  # [(2n + 1) * (2n + 1), B, T]

class Embedding(nn.Embedding):
    def forward_with_mask(self, indices: torch.LongTensor, valid_mask: torch.BoolTensor):
        # indices.shape = [B, T]
        # valid_mask.shape = [B, T]
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
            self.fwd = lambda x, valid_mask: super().forward(x)


    def forward(self, indices: torch.LongTensor, valid_mask = None):
        return self.fwd(indices, valid_mask)

class EventEmbedding(Embedding):
    def __init__(self, P: int, H: int, W: int, d: int):
        self.P = P
        self.H = H
        self.W = W
        self.d = d
        super().__init__(num_embeddings=P * H * W + 1, embedding_dim=d, padding_idx=0)

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask = None):
        indices = p * self.H * self.W + y * self.W + x
        return super().forward(indices, valid_mask)

class EventEmbedding2DCat(nn.Module):
    def __init__(self, P: int, H: int, W: int, d: int):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.d = d
        d_y = int(H / (H + W) * d)
        d_x = d - d_y
        assert d_y > 0 and d_x > 0

        self.embed_y = Embedding(num_embeddings=P * H + 1, embedding_dim=d_y, padding_idx=0)
        self.embed_x = Embedding(num_embeddings=P * W + 1, embedding_dim=d_x, padding_idx=0)


    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask = None):
        tokens_y = self.embed_y(p * self.H + y, valid_mask)
        tokens_x = self.embed_x(p * self.W + x, valid_mask)
        return torch.cat((tokens_y, tokens_x), dim=-1)

class WeightedNeighbor1DEventEmbedding(nn.Module):
    def __init__(self, P: int, H: int, W: int, d: int, n_neighbor:int, dilated:int=1):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.d = d

        self.n_neighbor = n_neighbor
        self.dilated = dilated
        self.neighbor_generator_y = Neighbor1D(L=H, n=n_neighbor, dilated=dilated)
        self.neighbor_generator_x = Neighbor1D(L=W, n=n_neighbor, dilated=dilated)


        c = torch.ones([2 * n_neighbor + 1])

        for dx in range(-n_neighbor, n_neighbor + 1):
            distance = dx * dx
            c[dx + n_neighbor] = math.exp(-distance / 2)

        c = c.flatten().view(-1, 1, 1, 1)
        self.c = nn.Parameter(c)

        self.embed = Embedding(num_embeddings=P * H * W + 1, embedding_dim=d, padding_idx=0)

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask=None):
        y = self.neighbor_generator_y(y)
        x = self.neighbor_generator_x(x)
        p = p.unsqueeze(0)
        tokens = self.embed(p * self.H * self.W + y * self.W + x, valid_mask.unsqueeze(0))
        return (tokens * self.c).sum(0)

class WeightedNeighbor1DEventEmbedding2DCat(nn.Module):
    def __init__(self, P: int, H: int, W: int, d: int, n_neighbor:int, dilated:int=1):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.d = d
        self.n_neighbor = n_neighbor
        self.dilated = dilated


        c_y = torch.ones(2 * n_neighbor + 1)
        for distance in range(-n_neighbor, n_neighbor + 1):
            c_y[distance + n_neighbor] = math.exp(-distance * distance / 2)

        self.c_y = nn.Parameter(c_y.view(-1, 1, 1, 1))

        c_x = torch.ones(2 * n_neighbor + 1)
        for distance in range(-n_neighbor, n_neighbor + 1):
            c_x[distance + n_neighbor] = math.exp(-distance * distance / 2)
        self.c_x = nn.Parameter(c_x.view(-1, 1, 1, 1))

        self.neighbor_generator_y = Neighbor1D(L=H, n=n_neighbor, dilated=dilated)
        self.neighbor_generator_x = Neighbor1D(L=W, n=n_neighbor, dilated=dilated)
        self.embed = EventEmbedding2DCat(P=P, H=H, W=W, d=d)




    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask=None):
        y = self.neighbor_generator_y(y)
        x = self.neighbor_generator_x(x)
        p = p.unsqueeze(0)
        tokens_y = self.embed.embed_y(p * self.H + y, valid_mask)
        tokens_x = self.embed.embed_x(p * self.W + x, valid_mask)
        tokens_y = (tokens_y * self.c_y).sum(0)
        tokens_x = (tokens_x * self.c_x).sum(0)
        return torch.cat((tokens_y, tokens_x), dim=-1)
    
    
class WeightedNeighbor2DEventEmbedding(nn.Module):
    def __init__(self, P: int, H: int, W: int, d: int, n_neighbor:int, dilated:int=1):
        super().__init__()
        self.P = P
        self.H = H
        self.W = W
        self.d = d

        self.n_neighbor = n_neighbor
        self.dilated = dilated
        self.neighbor_generator = Neighbor2D(H=H, W=W, n=n_neighbor, dilated=dilated)

        c = torch.ones([2 * n_neighbor + 1, 2 * n_neighbor + 1])

        for dy in range(-n_neighbor, n_neighbor + 1):
            for dx in range(-n_neighbor, n_neighbor + 1):
                distance = dy * dy + dx * dx
                c[dy + n_neighbor][dx + n_neighbor] = math.exp(-distance / 2)

        c = c.flatten().view(-1, 1, 1, 1)

        self.c = nn.Parameter(c)
        self.embed = Embedding(num_embeddings=P * H * W + 1, embedding_dim=d, padding_idx=0)

    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, valid_mask=None):
        coors = self.neighbor_generator(y, x)
        tokens = self.embed(p.unsqueeze(0) * self.H * self.W + coors, valid_mask.unsqueeze(0))
        return (tokens * self.c).sum(0)




class SpatialDownsample(nn.Module):
    def __init__(self, H: int, W: int, h: int, w: int):
        super().__init__()

        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.scale_y = h / H
        self.scale_x = w / W


    def extra_repr(self):
        return f'{self.H}x{self.W} -> {self.h}x{self.w}, {self.scale_y:.2f}x{self.scale_x:.2f}'

    def forward(self, y: torch.LongTensor, x: torch.LongTensor):
        y = (y * self.scale_y).round_().long()
        x = (x * self.scale_x).round_().long()
        torch.clamp_(y, 0, self.h - 1)
        torch.clamp_(x, 0, self.w - 1)
        return y, x


class AdditiveTemporalPositionalEncoding(nn.Module):
    def __init__(self, transform:nn.Module or Callable = lambda x: x.unsqueeze(2)):
        super().__init__()
        self.transform = transform

    def forward(self, t: torch.Tensor, tokens: torch.Tensor):
        # t.shape = [B, T]
        # tokens.shape = [B, T, d]
        return self.transform(t) + tokens

class Time2Vec(nn.Module):
    def __init__(self, d: int):
        # Time2Vec: Learning a Vector Representation of Time
        super().__init__()
        self.d = d
        self.fc = nn.Linear(1, d)

    def forward(self, t: torch.Tensor):
        t = self.fc(t.unsqueeze(2))
        t[:, :, 1:] = torch.sin(t[:, :, 1:].clone())
        return t



class MaskSequence(nn.Module):
    def __init__(self, p: float):
        # work in both training and inference
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        # x.shape = [B, T, d]
        B, T = x.shape[0], x.shape[1]
        mask = torch.rand([B, T, 1], device=x.device, dtype=x.dtype) > self.p
        return x * mask.to(x.dtype), mask

class DropSequence(nn.Module):
    def __init__(self, p: float):
        # only work in training
        super().__init__()
        self.p = p

        self.forwards = {
            True: self.training_forward,
            False: lambda x: x
        }

    def training_forward(self, x: torch.Tensor):
        # x.shape = [B, T, d]
        B, T = x.shape[0], x.shape[1]
        mask = torch.rand([B, T, 1], device=x.device, dtype=x.dtype) > self.p
        return x * mask.to(x.dtype)

    def forward(self, x: torch.Tensor):

        return self.forwards[self.training](x)


def create_transformer_encoder(d_model: int, n_head: int, d_feedforward: int, n_layers: int, dropout:float=0.):
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_feedforward,
                                               batch_first=True, dropout=dropout)
    return nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_feedforward: int, n_layers: int, n_classes: int, classify_head_zero_init:bool = False, dropout:float=0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_feedforward = d_feedforward
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))


        self.encoder = create_transformer_encoder(d_model=d_model, n_head=n_head, d_feedforward=d_feedforward, n_layers=n_layers, dropout=dropout)

        self.head = nn.Linear(d_model, n_classes)

        if classify_head_zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)


    def encoder_forward(self, tokens, valid_mask):
        # tokens.shape = [B, T, d]
        B = tokens.shape[0]
        batch_class_token = self.class_token.repeat(B, 1, 1)  # [B, 1, d]
        tokens = torch.cat([batch_class_token, tokens], dim=1)  # [B, T+1, d]
        padding_mask = torch.cat((
            torch.zeros([B, 1], dtype=valid_mask.dtype, device=valid_mask.device),
            ~valid_mask
        ), dim=1)  # [B, T+1]

        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return tokens

    def forward(self, tokens, valid_mask):
        tokens = self.encoder_forward(tokens, valid_mask)

        tokens = tokens[:, 0]

        tokens = self.head(tokens)

        return tokens




class MergeTokenHead(nn.Module):
    def __init__(self, t_heads: int):
        super().__init__()
        self.t_heads = t_heads

    def extra_repr(self) -> str:
        return f't_heads={self.t_heads}'

    def forward(self, tokens: torch.Tensor, valid_mask: torch.BoolTensor):
        # shape = [B, T, d] -> [B, t_heads, T // t_heads, d]
        B = tokens.shape[0]
        d = tokens.shape[2]
        tokens = tokens.view(B, self.t_heads, -1, d)
        valid_mask = valid_mask.view(B, self.t_heads, -1, 1)
        valid_mask = valid_mask.to(tokens.dtype)


        tokens = tokens * valid_mask
        reduced_valid_mask = valid_mask.sum(1)  # [B, T // t_heads, 1]
        tokens = tokens.sum(1) / reduced_valid_mask

        return tokens, reduced_valid_mask.squeeze(2) >= 1

class MergeTimeHead(nn.Module):
    def __init__(self, t_heads: int):
        super().__init__()
        self.t_heads = t_heads

    def extra_repr(self) -> str:
        return f't_heads={self.t_heads}'

    def forward(self, t: torch.Tensor, valid_mask: torch.BoolTensor):
        # shape = [B, T] -> [B * t_heads, T // t_heads]
        B = t.shape[0]
        t = t.view(B, self.t_heads, -1)
        valid_mask = valid_mask.view(B, self.t_heads, -1)
        valid_mask = valid_mask.to(t.dtype)

        t = (t * valid_mask).sum(1) / valid_mask.sum(1)
        return t

class CoordinateNoise(nn.Module):
    def __init__(self, std: float, H: int, W: int):
        super().__init__()
        self.std = std
        self.H = H
        self.W = W
        self.forwards = {
            True: self.training_forward,
            False: lambda y, x: (y, x),
        }

    def forward(self, y: torch.LongTensor, x: torch.LongTensor):
        return self.forwards[self.training](y, x)

    def training_forward(self, y: torch.LongTensor, x: torch.LongTensor):
        # shape = [B, L]
        B, L = y.shape
        noise = (torch.randn([2], device=y.device) * self.std).round_().to(y.dtype)
        y += noise[0]
        x += noise[1]
        torch.clamp_(y, 0, self.H - 1)
        torch.clamp_(x, 0, self.W - 1)
        return y, x



class Event2VecClassifier(lightning.LightningModule):
    @staticmethod
    def create_embedding_layer(embed: str, P:int, H: int, W: int, d_model: int):
        if embed == 'naive':
            return EventEmbedding(P=P, H=H, W=W, d=d_model)
        elif embed == 'naive2d':
            return EventEmbedding2DCat(P=P, H=H, W=W, d=d_model)
        elif embed.startswith('neighbor1d-naive-'):
            # neighbor1d-naive-{n_neighbor}-{dilated}
            embed = embed.split('-')
            n_neighbor = int(embed[2])
            dilated = int(embed[3])
            return WeightedNeighbor1DEventEmbedding(P=P, H=H, W=W, d=d_model, n_neighbor=n_neighbor,
                                                          dilated=dilated)
        elif embed.startswith('neighbor1d-naive2d-'):
            # neighbor1d-naive2d-{n_neighbor}-{dilated}
            embed = embed.split('-')
            n_neighbor = int(embed[2])
            dilated = int(embed[3])
            return WeightedNeighbor1DEventEmbedding2DCat(P=P, H=H, W=W, d=d_model, n_neighbor=n_neighbor,
                                                               dilated=dilated)
        elif embed.startswith('neighbor2d-naive-'):
            # neighbor2d-naive-{n_neighbor}-{dilated}
            embed = embed.split('-')
            n_neighbor = int(embed[2])
            dilated = int(embed[3])
            return WeightedNeighbor2DEventEmbedding(P=P, H=H, W=W, d=d_model, n_neighbor=n_neighbor,
                                                          dilated=dilated)



        else:
            raise NotImplementedError(embed)
    def __init__(self, P: int, H: int, W: int, h: int, w: int, embed:str, tpe: str, d_model: int, n_head: int, d_feedforward: int, n_layers: int, t_heads:int, n_classes: int, classify_head_zero_init:bool = False, dropout:float=0., dropout_sequence:float=0.,
                 offset_coordinate:bool=False, offset_t:bool=False,
                 embed_pretrained:str=None, pretrained:str=None, learn_head_only:bool=False,
                 self_supervised:bool=False,
                 compile:bool = False, lr:float = 1e-3, optimizer:str='adamw', wd: float=0., lrs:str='CosineAnnealingLR', label_smoothing:float=0., coordinate_noise_std:float=0.):
        '''
        accoding to https://kexue.fm/archives/7695
        # n > 8.33 * log(N), where n is the embedding dim and N is the vocabulary size
        # for DVS 128 camera, n > 37.61 is enough?
        '''
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer.lower()
        self.wd = wd
        self.lrs = lrs.lower()
        self.label_smoothing = label_smoothing
        self.coordinate_noise_std = coordinate_noise_std
        self.dropout_sequence = dropout_sequence

        self.P = P
        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.t_heads = t_heads

        if h < H or w < W:
            self.spatial_downsample = SpatialDownsample(H=H, W=W, h=h, w=w)
        else:
            self.spatial_downsample = lambda y, x: (y, x)

        del H, W
        if coordinate_noise_std > 0:
            self.coordinate_noise = CoordinateNoise(std=coordinate_noise_std, H=h, W=w)
        else:
            self.coordinate_noise = lambda y, x: (y, x)

        if t_heads == 1:
            self.merge_token_head = lambda tokens, valid_mask: (tokens, valid_mask)
            self.merge_t_head = lambda t, valid_mask: t
        else:
            self.merge_token_head = MergeTokenHead(t_heads=t_heads)
            self.merge_t_head = MergeTimeHead(t_heads=t_heads)


        self.embed = self.create_embedding_layer(embed, P, h, w, d_model)

        if embed_pretrained is not None:
            sd = torch.load(embed_pretrained, map_location='cpu')['state_dict']
            embed_sd = OrderedDict()
            for k in sd.keys():
                if k.startswith('embed'):
                    v = sd[k]
                    embed_sd[''.join(k.split('.')[1:])] = v
            self.embed.load_state_dict(embed_sd)
            print('self.embed.load_state_dict', embed_sd.keys())
            del sd, embed_sd


        if tpe == 'add':
            self.tpe = AdditiveTemporalPositionalEncoding(transform=lambda x: x.unsqueeze(2))
        elif tpe == 't2v':
            self.tpe = AdditiveTemporalPositionalEncoding(transform=Time2Vec(d_model))

        else:
            raise NotImplementedError(tpe)

        if self_supervised:
            assert dropout_sequence > 0
            self.mask_seq = MaskSequence(dropout_sequence)
        else:
            if dropout_sequence > 0:
                self.mask_seq = DropSequence(dropout_sequence)
            else:
                self.mask_seq = lambda x: x

        self.classifier = TransformerEncoderClassifier(d_model=d_model, n_head=n_head, d_feedforward=d_feedforward, n_layers=n_layers, n_classes=n_classes, classify_head_zero_init=classify_head_zero_init, dropout=dropout)

        if self_supervised:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.embed.num_embeddings, ignore_index=-1)
            self.valid_acc = Accuracy(task="multiclass", num_classes=self.embed.num_embeddings, ignore_index=-1)
            self.test_acc = Accuracy(task="multiclass", num_classes=self.embed.num_embeddings, ignore_index=-1)

        else:
            self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
            self.valid_acc = Accuracy(task="multiclass", num_classes=n_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)


        self.compile_flag = compile
        self.print_info = ''
        self.train_duration = 0.
        self.valid_duration = 0.

        if pretrained:
            sd = torch.load(pretrained, map_location='cpu')['state_dict']
            # classify head will not be loaded
            removed_keys = []
            for k in sd.keys():
                if k.startswith('classifier.head'):
                    removed_keys.append(k)
            for k in removed_keys:
                sd.pop(k)

            self.load_state_dict(sd, strict=False)

        self.learn_head_only = learn_head_only
        if learn_head_only:
            for p in self.parameters():
                p.requires_grad = False

            for p in self.classifier.head.parameters():
                p.requires_grad = True

        self.self_supervised = self_supervised
        # mask ratio is determined by dropout_sequence



        if offset_coordinate:
            self.offset_coordinate = OffsetCoordinate()
        else:
            self.offset_coordinate = lambda p, y, x, valid_mask: (p, y, x, valid_mask)

        if offset_t:
            self.offset_t = OffsetTime()
        else:
            self.offset_t = lambda t: t


        self.forwards = {
            False: self.classify_forward,
            True: self.self_supervised_forward_ce
        }

        print(self)

    def training_step(self, batch, batch_idx):
        xytp, valid_mask, label = batch
        self.train_samples += label.shape[0]
        if self.self_supervised:
            indices, target = self(xytp, valid_mask)
            loss = F.cross_entropy(indices, target, ignore_index=-1, label_smoothing=self.label_smoothing)
            self.train_acc.update(indices, target)
            # loss = self(xytp, valid_mask)

        else:
            label_predicted = self(xytp, valid_mask)
            loss = F.cross_entropy(label_predicted, label, label_smoothing=self.label_smoothing)
            self.train_acc.update(label_predicted, label)
        self.log_dict({"train_loss": loss}, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xytp, valid_mask, label = batch
        self.val_samples += label.shape[0]
        if self.self_supervised:
            indices, target = self(xytp, valid_mask)
            loss = F.cross_entropy(indices, target, ignore_index=-1, label_smoothing=self.label_smoothing)
            self.valid_acc.update(indices, target)
            # loss = self(xytp, valid_mask)



        else:
            label_predicted = self(xytp, valid_mask)
            loss = F.cross_entropy(label_predicted, label)
            self.valid_acc.update(label_predicted, label)
        self.log_dict({"valid_loss": loss}, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch_device_L, batch_idx):
        assert not self.self_supervised
        batch, device, L = batch_device_L
        xytp, valid_mask, label = batch
        label = label.to(device, non_blocking=True)
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]

        x = [x[:, i::L] for i in range(L)]
        y = [y[:, i::L] for i in range(L)]
        t = [t[:, i::L] for i in range(L)]
        p = [p[:, i::L] for i in range(L)]
        valid_mask = [valid_mask[:, i::L] for i in range(L)]
        n = len(x)

        self.test_samples += label.shape[0]

        for i in range(n):
            print(f'\rprocessing {i}/{len(x)}, {batch_idx}/{self.test_batch_number}', end='')
            sub_x = x.pop().to(device, non_blocking=True)
            sub_y = y.pop().to(device, non_blocking=True)
            sub_t = t.pop().to(device, non_blocking=True)
            sub_p = p.pop().to(device, non_blocking=True)
            sub_valid_mask = valid_mask.pop().to(device, non_blocking=True)
            if i == 0:
                label_predicted = self((sub_x, sub_y, sub_t, sub_p), sub_valid_mask)
            else:
                label_predicted += self((sub_x, sub_y, sub_t, sub_p), sub_valid_mask)

        self.test_acc.update(label_predicted, label)

    def on_test_epoch_start(self) -> None:
        self.test_samples = 0
        self.test_start_time = time.time()
        self.test_batch_number = len(self.trainer.test_dataloaders)

    def on_test_epoch_end(self) -> None:
        test_acc = self.test_acc.compute()
        self.test_acc.reset()
        self.test_end_time = time.time()
        self.test_duration = self.test_end_time - self.test_start_time
        self.test_speed = self.test_samples / self.test_duration
        print('\n')
        print(f'test_acc={test_acc:.3f}, test_speed={self.test_speed:.3f} samples/sec')

    def on_validation_epoch_start(self):
        self.val_samples = 0
        self.valid_start_time = time.time()


    def on_validation_epoch_end(self):
        valid_acc = self.valid_acc.compute()
        epoch_metrics = self.trainer.callback_metrics
        self.log('valid_acc', valid_acc)
        self.valid_acc.reset()
        self.valid_end_time = time.time()
        self.valid_duration = self.valid_end_time - self.valid_start_time
        self.valid_speed = self.val_samples / self.valid_duration
        self.val_samples = 0
        print(f'valid_loss={epoch_metrics["valid_loss"]:.3f}, valid_acc={valid_acc:.3f}, valid_speed={self.valid_speed:.3f} samples/sec')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(self.train_duration + self.valid_duration) * (self.trainer.max_epochs - self.current_epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')



    def on_train_epoch_start(self):
        self.train_samples = 0
        self.train_start_time = time.time()


    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        epoch_metrics = self.trainer.callback_metrics
        print(self.print_info)
        self.log('train_acc', train_acc)
        self.train_acc.reset()
        self.train_end_time = time.time()
        self.train_duration = self.train_end_time - self.train_start_time
        self.train_speed = self.train_samples / self.train_duration
        self.train_samples = 0
        print(f'epoch={self.current_epoch}, train_loss={epoch_metrics["train_loss"]:.3f}, train_acc={train_acc:.3f}, train_speed={self.train_speed:.3f} samples/sec')


    def self_supervised_forward_ce(self, xytp, valid_mask):
        # used for computing ce loss
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]


        y, x = self.spatial_downsample(y, x)
        y, x = self.coordinate_noise(y, x)

        p, y, x, valid_mask = self.offset_coordinate(p, y, x, valid_mask)
        t = self.offset_t(t)

        t = self.merge_t_head(t, valid_mask)

        tokens = self.embed(p, y, x, valid_mask)

        tokens, valid_mask = self.merge_token_head(tokens, valid_mask)

        tokens, mask = self.mask_seq(tokens)

        tokens = self.tpe(t, tokens)

        # note that tokens[:, 0] is the cls
        tokens = self.classifier.encoder_forward(tokens, valid_mask)[:, 1:]


        indices = tokens @ self.embed.weight.T  # [B, L, d_embed]

        # note that indices +=1 in Embedding
        # so, we also need to +1 here
        target = p * self.embed.H * self.embed.W + y * self.embed.W + x  # [B, L]

        target += 1
        target *= valid_mask
        padding_mask = ~valid_mask
        target[padding_mask] = -1


        mask = ~mask

        indices = indices[mask.repeat(1, 1, indices.shape[2])]
        target = target[mask.squeeze(2)]
        indices = indices.view(-1, self.embed.num_embeddings)

        return indices, target

    def self_supervised_forward_mse(self, xytp, valid_mask):
        # used for computingg mse loss
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]

        y, x = self.spatial_downsample(y, x)
        y, x = self.coordinate_noise(y, x)

        p, y, x, valid_mask = self.offset_coordinate(p, y, x, valid_mask)
        t = self.offset_t(t)



        t = self.merge_t_head(t, valid_mask)

        tokens = self.embed(p, y, x, valid_mask)

        tokens, valid_mask = self.merge_token_head(tokens, valid_mask)

        tokens_ = tokens.clone()
        tokens, mask = self.mask_seq(tokens)

        tokens = self.tpe(t, tokens)

        padding_mask = ~valid_mask
        tokens = self.classifier.encoder(tokens, src_key_padding_mask=padding_mask)

        mask = ~mask
        mask = mask.repeat(1, 1, tokens.shape[2])
        return F.mse_loss(tokens[mask], tokens_[mask])


    def classify_forward(self, xytp, valid_mask):
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]



        y, x = self.spatial_downsample(y, x)
        y, x = self.coordinate_noise(y, x)
        p, y, x, valid_mask = self.offset_coordinate(p, y, x, valid_mask)
        t = self.offset_t(t)


        t = self.merge_t_head(t, valid_mask)

        tokens = self.embed(p, y, x, valid_mask)



        tokens, valid_mask = self.merge_token_head(tokens, valid_mask)
        tokens = self.mask_seq(tokens)

        tokens = self.tpe(t, tokens)


        return self.classifier(tokens, valid_mask)

    def forward(self, xytp, valid_mask):
        return self.forwards[self.self_supervised](xytp, valid_mask)

    def configure_optimizers(self):
        if self.wd > 0:
            decay_params = set()
            no_decay_params = set()
            processed_params = set()
            for module_name, module in self.named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                    processed_params.add(full_param_name)

                    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.Embedding)) or 'class_token' in full_param_name:
                        no_decay_params.add(param)
                    elif param_name == 'bias':
                        no_decay_params.add(param)
                    else:
                        decay_params.add(param)

            for name, param in self.named_parameters():
                if name not in processed_params:
                    decay_params.add(param)


            parameters = [
                {"params": list(decay_params), "weight_decay": self.wd},
                {"params": list(no_decay_params), "weight_decay": 0.}
            ]
        else:
            parameters = self.parameters()

        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.wd, fused=True)
        else:
            raise NotImplementedError(self.optimizer_name)
        if self.lrs == 'cosineannealinglr':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        elif self.lrs == 'none':
            lr_scheduler = None
        else:
            raise NotImplementedError(self.lrs)

        if lr_scheduler is None:
            return optimizer
        else:
            return ([optimizer], [lr_scheduler])

