import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import event_embedding, temporal_pe

import utils
from model_zoo import fox_encoder
import event_transform

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@torch.compile
def sequence_avg_pooling(tokens: torch.Tensor, valid_mask: torch.BoolTensor, stride: int):
    B, L, d = tokens.shape
    if L % stride != 0:
        print('L % stride != 0')
        print(tokens.shape, stride)
        exit(-1)
    valid_mask = valid_mask.view(B, L // stride, stride).float()
    tokens = tokens.view(B, L // stride, stride, d)
    valid_mask_sum = valid_mask.sum(2)
    tokens = (tokens * valid_mask.unsqueeze(3)).sum(2) / (valid_mask_sum.unsqueeze(2) + 1e-5)
    valid_mask = valid_mask_sum > 0
    return tokens, valid_mask



class E2VNet(nn.Module):
    def __init__(self, P: int, H: int, W: int, d_model: int, d_feedforward: int, nheads: int, n_layers: int,
                 n_classes: int, activation: str, mask_ratio:float, p_token_mix:float=0., p_intensity_drop:float=0., drop_path:float=0., pool_every_layer:int=0):
        super().__init__()

        self.self_supervised_training = mask_ratio > 0

        self.P = P
        self.H = H
        self.W = W
        self.norm_type = 'ln'


        if self.norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif self.norm_type == 'rms':
            norm_class = nn.RMSNorm



        self.embed_net = event_embedding.MLPEmbedding(P=P, H=H, W=W, d_model=d_model, norm=False,
                                                      return_embed_matrix=False, norm_type=self.norm_type, activation=activation)
        self.tpe = temporal_pe.Conv(kernel_size=3, d=d_model, norm_type=self.norm_type, activation=activation)

        if self.self_supervised_training:
            self.mask_ratio = mask_ratio
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)

            self.heads = nn.Sequential(
                # 与 MLPEmbedding 的最后一层对称
                nn.Linear(d_model, d_model // 2, bias=False),
                norm_class(d_model // 2),
                utils.create_activation(activation, inplace=True),

                # 与 MLPEmbedding 的中间一层对称
                nn.Linear(d_model // 2, d_model // 4, bias=False),
                norm_class(d_model // 4),
                utils.create_activation(activation, inplace=True),

                # 与 MLPEmbedding 的第一层对称，输出为 3
                nn.Linear(d_model // 4, 3, bias=False)
            )



        self.pool_every_layer = pool_every_layer
        if pool_every_layer > 0:
            self.backbone_net = _get_clones(fox_encoder.create_fox_encoder(d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                                    n_layers=pool_every_layer, activation=activation,
                                                    ffn_dropout=0., bias=False, norm=norm_class(d_model), drop_path=drop_path,
                                                    norm_type=self.norm_type, pre_norm=False), n_layers // pool_every_layer)
        else:
            self.backbone_net = fox_encoder.create_fox_encoder(d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                                    n_layers=n_layers, activation=activation,
                                                    ffn_dropout=0., bias=False, norm=norm_class(d_model), drop_path=drop_path,
                                                    norm_type=self.norm_type, pre_norm=False)

        if not self.self_supervised_training:
            self.heads = nn.Linear(d_model, n_classes)
            self.p_token_mix = p_token_mix
            if self.p_token_mix > 0:
                self.mixer = event_transform.TokenMix(num_classes=n_classes)
            self.p_intensity_drop = p_intensity_drop


    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor,
                t: torch.LongTensor, intensity, valid_mask=None, targets=None):
        if self.self_supervised_training:
            return self.self_supervised_bert_forward(p, y, x, t, intensity, valid_mask, targets)
        else:
            return self.classify_forward(p, y, x, t, intensity, valid_mask, targets)


    def self_supervised_bert_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor,
                                     t: torch.LongTensor, intensity, valid_mask=None, targets=None):

        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.

        valid_mask_f = valid_mask.float()
        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]
        t = torch.cat((torch.zeros_like(t[:, 0:1]), t), dim=1)
        tokens = self.embed_net(p, y, x, valid_mask)
        tpe = self.tpe(t) * valid_mask_f.unsqueeze(2)
        if intensity is not None:
            intensity = intensity.to(tokens)

        B, L = valid_mask.shape

        mask = torch.rand([B, L], device=t.device) < self.mask_ratio
        mask = mask & valid_mask

        x_mask = x[mask]
        y_mask = y[mask]
        p_mask = p[mask]
        if intensity is not None:
            intensity_mask = intensity[mask]
        tpe_mask = tpe[mask]

        tokens[mask] = self.mask_token.to(tokens)
        if intensity is not None:
            tokens = (tokens + tpe) * intensity.unsqueeze(2)
        else:
            tokens = tokens + tpe
        if self.pool_every_layer > 0:
            raise NotImplementedError
        
        tokens = self.backbone_net(tokens, valid_mask)
        if intensity is not None:
            predicted_embeddings = tokens[mask] / intensity_mask.unsqueeze(1) - tpe_mask
        else:
            predicted_embeddings = tokens[mask] - tpe_mask

        predicts = F.tanh(self.heads(predicted_embeddings))
        p_predicted = predicts[..., 0]
        y_predicted = predicts[..., 1]
        x_predicted = predicts[..., 2]

        loss_p = F.mse_loss(p_predicted, p_mask, reduction='sum')
        loss_y = F.mse_loss(y_predicted, y_mask, reduction='sum')
        loss_x = F.mse_loss(x_predicted, x_mask, reduction='sum')

        n_mask = mask.long().sum()
        loss = (loss_p + loss_y + loss_x) / n_mask
        with torch.no_grad():
            true_y_pixel = ((y_mask + 1.) / 2. * (self.H - 1)).round().long()
            true_x_pixel = ((x_mask + 1.) / 2. * (self.W - 1)).round().long()
            true_p_final = ((p_mask + 1.) / 2.).round().long()

            metrics = utils.calculate_event_prediction_metrics(pred_p_logit=p_predicted,
                                                               pred_y_norm=y_predicted,
                                                               pred_x_norm=x_predicted,
                                                               true_p=true_p_final,
                                                               true_y=true_y_pixel,
                                                               true_x=true_x_pixel,
                                                               height=self.H,
                                                               width=self.W,
                                                               valid_mask=None,
                                                               neighbors=3)

        return (n_mask.item(), metrics, loss), targets

    def classify_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, t: torch.LongTensor,
                         intensity,
                         valid_mask=None, targets=None):

        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.

        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        valid_mask_f = valid_mask.float()
        t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]

        t = torch.cat((torch.zeros_like(t[:, 0:1]), t), dim=1)

        tokens = self.embed_net(p, y, x, valid_mask)

        if intensity is not None:
            if self.training and self.p_intensity_drop > 0:
                mask = torch.rand_like(intensity) < self.p_intensity_drop
                intensity[mask] = 1
            tokens = (tokens + self.tpe(t)) * (intensity.to(tokens) * valid_mask_f).unsqueeze(2)
        else:
            tokens = (tokens + self.tpe(t)) * valid_mask_f.unsqueeze(2)

        if self.training and self.p_token_mix > 0:
            if torch.rand(size=[1]).item() < self.p_token_mix:
                tokens, targets = self.mixer(tokens, targets)


        if self.pool_every_layer > 0:
            for i in range(len(self.backbone_net)):
                tokens = self.backbone_net[i](tokens, valid_mask)
                tokens, valid_mask = sequence_avg_pooling(tokens=tokens, valid_mask=valid_mask, stride=2)
        else:
            tokens = self.backbone_net(tokens, valid_mask)

        valid_mask_f = valid_mask.float()
       
        tokens = (tokens * valid_mask_f.unsqueeze(2)).sum(1) / (valid_mask_f.sum(dim=1, keepdim=True) + 1e-5)

        tokens = self.heads(tokens)
        return tokens, targets
