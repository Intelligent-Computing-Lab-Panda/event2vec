import torch
import torch.nn as nn
import torch.nn.functional as F
from model_zoo import gla, common
import event_embedding



class DVSGestureNet(nn.Module):
    def __init__(self, P:int, H: int, W: int, d_model:int, d_feedforward: int, nheads: int, n_classes: int, activation: str):
        super().__init__()


        self.P = P
        self.H = H
        self.W = W


        self.embed_net = event_embedding.ParametricEmbedding(P=P, H=H, W=W, d_model=d_model, return_embed_matrix=False)
        self.tpe = event_embedding.TimestampConv(kernel_size=3, d=d_model)




        self.backbone_net = nn.ModuleList()
        self.pool_size = [2, 2, 2, 2, 2, 2, 2]
        self.stages = len(self.pool_size)
        n_layers = [1] * self.stages
        expand_k = expand_v = 0.5
        for i in range(self.stages):
            self.backbone_net.append(gla.create_gla_encoder(d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                   n_layers=n_layers[i], activation=activation, conv_size=0,
                                   bidirectional=True, expand_k=expand_k, expand_v=expand_v, ffn_type='plain', ffn_dropout=0, bias=False))




        self.heads_dropout = nn.Dropout(0)
        self.heads = nn.Linear(d_model, n_classes)



    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor,
                                t: torch.LongTensor, valid_mask=None):

        return self.classify_forward(p, y, x, t, valid_mask)



    def classify_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, t: torch.LongTensor, valid_mask=None):
        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        valid_mask_f = valid_mask.float()
        diff_t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]

        diff_t = torch.cat((torch.zeros_like(t[:, 0:1]), diff_t), dim=1)

        tokens = self.embed_net(p, y, x, valid_mask)

        tokens = tokens + self.tpe(diff_t) * valid_mask_f.unsqueeze(2)

        for i in range(self.stages):
            tokens, valid_mask = common.sequence_avg_pooling(tokens, valid_mask, self.pool_size[i], avg=True)
            tokens = self.backbone_net[i](tokens, valid_mask)

            # if i + 1 < self.pools:
            #     tokens, valid_mask = common.sequence_avg_pooling(tokens, valid_mask, self.pool_size[i], avg=True)
            #     # tokens, valid_mask = common.sequence_max_pooling(tokens, valid_mask, self.pool_size[i])

        valid_mask_f = valid_mask.float()

        tokens = (tokens * valid_mask_f.unsqueeze(2)).sum(1) / (valid_mask_f.sum(dim=1, keepdim=True) + 1e-5)


        tokens = self.heads_dropout(tokens)
        tokens = self.heads(tokens)

        return tokens

class ASLDVSNet(nn.Module):
    def __init__(self, P:int, H: int, W: int, d_model:int, d_feedforward: int, nheads: int, n_classes: int, activation: str):
        super().__init__()


        self.P = P
        self.H = H
        self.W = W


        self.embed_net = event_embedding.ParametricEmbedding(P=P, H=H, W=W, d_model=d_model, return_embed_matrix=False)
        self.tpe = event_embedding.TimestampConv(kernel_size=3, d=d_model)




        self.backbone_net = nn.ModuleList()
        self.pool_size = [1]
        self.stages = len(self.pool_size)
        n_layers = [2] * self.stages
        expand_k = expand_v = 1
        for i in range(self.stages):
            self.backbone_net.append(gla.create_gla_encoder(d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                   n_layers=n_layers[i], activation=activation, conv_size=0,
                                   bidirectional=True, expand_k=expand_k, expand_v=expand_v, ffn_type='plain', ffn_dropout=0, bias=False))




        self.heads_dropout = nn.Dropout(0)
        self.heads = nn.Linear(d_model, n_classes)



    def forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor,
                                t: torch.LongTensor, valid_mask=None):

        return self.classify_forward(p, y, x, t, valid_mask)



    def classify_forward(self, p: torch.LongTensor, y: torch.LongTensor, x: torch.LongTensor, t: torch.LongTensor, valid_mask=None):
        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        valid_mask_f = valid_mask.float()
        diff_t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]

        diff_t = torch.cat((torch.zeros_like(t[:, 0:1]), diff_t), dim=1)

        tokens = self.embed_net(p, y, x, valid_mask)

        tokens = tokens + self.tpe(diff_t) * valid_mask_f.unsqueeze(2)

        for i in range(self.stages):
            if self.pool_size[i] > 1:
                tokens, valid_mask = common.sequence_avg_pooling(tokens, valid_mask, self.pool_size[i], avg=True)
            tokens = self.backbone_net[i](tokens, valid_mask)

            # if i + 1 < self.pools:
            #     tokens, valid_mask = common.sequence_avg_pooling(tokens, valid_mask, self.pool_size[i], avg=True)
            #     # tokens, valid_mask = common.sequence_max_pooling(tokens, valid_mask, self.pool_size[i])

        valid_mask_f = valid_mask.float()

        tokens = (tokens * valid_mask_f.unsqueeze(2)).sum(1) / (valid_mask_f.sum(dim=1, keepdim=True) + 1e-5)


        tokens = self.heads_dropout(tokens)
        tokens = self.heads(tokens)

        return tokens
