import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import event_embedding, temporal_pe

import utils
from model_zoo import fla_encoder
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
    def __init__(self, P: int, H: int, W: int, attention:str, d_model: int, d_feedforward: int, nheads: int, n_layers: int,
                 n_classes: int, activation: str, mask_ratio:float, mask_len:int,
                 p_token_mix:float=0., p_intensity_drop:float=0., drop_path:float=0., pool_every_layer:int=0,
                 bi_share_param:bool=True, intensity_norm:str='none',
                 contrastive_learning_temperature:float=-1,
                 self_supervised_style:str='bert',
                 spatial_embed:str='plain', temporal_embed:str='plain',
                 embed_fusion:str='plain'
                 ):
        super().__init__()

        assert not (mask_ratio > 0 and contrastive_learning_temperature > 0)

        self.self_supervised_training = mask_ratio > 0
        self.self_supervised_style = self_supervised_style

        self.contrastive_learning_temperature = contrastive_learning_temperature
        if contrastive_learning_temperature > 0:
            # 启用对比学习
            self.contrastive_learning_loss = utils.SupConLoss()

        self.P = P
        self.H = H
        self.W = W
        self.norm_type = 'ln'


        if self.norm_type == 'ln':
            norm_class = nn.LayerNorm
        elif self.norm_type == 'rms':
            norm_class = nn.RMSNorm

        self.intensity_norm = intensity_norm
        if spatial_embed == 'plain':
            self.embed_net = event_embedding.MLPEmbedding(d_model=d_model, norm_type=self.norm_type, activation=activation, in_features=3)
        elif spatial_embed == 'fourier':
            self.embed_net = event_embedding.MLPEmbedding(d_model=d_model, norm_type=self.norm_type, activation=activation, in_features=3)

        if temporal_embed == 'plain':
            self.tpe = temporal_pe.Conv(kernel_size=3, d=d_model, norm_type=self.norm_type, activation=activation)
        elif temporal_embed == 'fourier':
            self.tpe = temporal_pe.FourierTemporalEmbedding(kernel_size=3, d=d_model, norm_type=self.norm_type, activation=activation)
        
        self.embed_fusion = embed_fusion
        if embed_fusion == 'plain':
            pass
        elif embed_fusion == 'gated':
            self.embed_fusion_net = event_embedding.GatedSpatioTemporalFusion(d_model=d_model)
        else:
            raise NotImplementedError
        if self.self_supervised_training:
            self.mask_ratio = mask_ratio
            self.mask_len = mask_len
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)

            if self_supervised_style == 'mae':
                self.enc_to_dec_proj = nn.Linear(d_model, 192) if d_model != 192 else nn.Identity()
                self.decoder_net = fla_encoder.create_fla_encoder(attention=attention, d_model=192, num_heads=6, d_feedforward=384,
                                                    n_layers=2, activation=activation,
                                                    ffn_dropout=0., bias=False, norm=norm_class(d_model), drop_path=0.,
                                                    norm_type=self.norm_type, pre_norm=False, bi_share_param=bi_share_param)

            self.heads = nn.Sequential(
                # 与 MLPEmbedding 的最后一层对称
                nn.Linear(d_model, d_model // 2, bias=False),
                norm_class(d_model // 2),
                utils.create_activation(activation),

                # 与 MLPEmbedding 的中间一层对称
                nn.Linear(d_model // 2, d_model // 4, bias=False),
                norm_class(d_model // 4),
                utils.create_activation(activation),

                # 与 MLPEmbedding 的第一层对称，输出为 3
                nn.Linear(d_model // 4, 3, bias=False)
            )



        self.pool_every_layer = pool_every_layer
        if pool_every_layer > 0:
            self.backbone_net = _get_clones(fla_encoder.create_fla_encoder(attention=attention, d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                                    n_layers=pool_every_layer, activation=activation,
                                                    ffn_dropout=0., bias=False, norm=norm_class(d_model), drop_path=drop_path,
                                                    norm_type=self.norm_type, pre_norm=False, bi_share_param=bi_share_param), n_layers // pool_every_layer)
        else:
            self.backbone_net = fla_encoder.create_fla_encoder(attention=attention, d_model=d_model, num_heads=nheads, d_feedforward=d_feedforward,
                                                    n_layers=n_layers, activation=activation,
                                                    ffn_dropout=0., bias=False, norm=norm_class(d_model), drop_path=drop_path,
                                                    norm_type=self.norm_type, pre_norm=False, bi_share_param=bi_share_param)

        if not self.self_supervised_training:
            if self.contrastive_learning_temperature > 0:
                self.heads = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(d_model * 4, 128) 
                )
            else:
                self.heads = nn.Linear(d_model, n_classes)
            self.p_token_mix = p_token_mix
            if self.p_token_mix > 0:
                self.mixer = event_transform.TokenMix(num_classes=n_classes)
            self.p_intensity_drop = p_intensity_drop


    def forward(self, batch):
        if self.self_supervised_training:
            return self.self_supervised_bert_forward(batch)
        else:
            return self.classify_forward(batch)


    def norm_intensity_fun(self, intensity):
        if self.intensity_norm == 'none':
            return intensity
        elif self.intensity_norm == 'log':
            intensity.clamp_min_(1)
            return torch.log(intensity) + 1.
        else:
            raise NotImplementedError
    
    def self_supervised_bert_forward(self, batch):

        p = batch['p']
        y = batch['y']
        x = batch['x']
        t = batch['t']
        if 'intensity' in batch:
            intensity = batch['intensity']
        else:
            intensity = None
                          
        valid_mask = batch['valid_mask']


        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.


        if intensity is not None:
            intensity = intensity.to(x)
            
            intensity = self.norm_intensity_fun(intensity)

        valid_mask_f = valid_mask.float()
        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]
        t = torch.cat((torch.zeros_like(t[:, 0:1]), t), dim=1)


        tokens = self.embed_net(torch.stack((x, y, p), dim=2), valid_mask)

        tpe = self.tpe(t) * valid_mask_f.unsqueeze(2)



        B, L = valid_mask.shape
        if self.mask_len == 1:
            mask = torch.rand([B, L], device=t.device) < self.mask_ratio
        else:
            # --- 1. 根据 self.mask_len 动态计算参数 ---
            # 几何分布的期望是 1/p。如果你希望平均长度是 self.mask_len，
            # 那么 p = 1 / self.mask_len
            prob = 1.0 / self.mask_len
            
            # 设定最大截断长度。通常取平均长度的 2 倍或固定值 10，防止太长
            # 这里设定为平均长度的 2 倍，最少为 10
            span_max_len = max(10, int(self.mask_len * 2))
            
            # --- 2. 计算起始点概率 ---
            # 为了让最终被 mask 的总比例接近 mask_ratio
            # 起始点的概率 = 总 mask 比例 / 每个 span 的平均长度
            start_prob = self.mask_ratio / self.mask_len
            
            # --- 3. 采样起始点 ---
            start_mask = torch.rand([B, L], device=t.device) < start_prob
            
            # --- 4. 采样每个起始点的长度 (服从几何分布) ---
            # PyTorch 的 Geometric 采样结果是 0, 1, 2... (表示失败次数)
            # 我们需要长度至少为 1，所以 +1。这样 E[len] = 1/p = self.mask_len
            span_lens = torch.distributions.geometric.Geometric(probs=prob).sample([B, L]).to(t.device) + 1
            span_lens = span_lens.clamp(max=span_max_len).int() # 截断并转为整数
            
            # --- 5. 生成最终 Mask ---
            mask = torch.zeros([B, L], dtype=torch.bool, device=t.device)
            
            # 只需要循环到最大截断长度即可，通常很小（比如 10-15 次），速度很快
            for i in range(span_max_len):
                # 找出那些“长度足够覆盖当前偏移量 i”的起始点
                active_starts = start_mask & (span_lens > i)
                
                if i == 0:
                    mask = mask | active_starts
                else:
                    # 向量化向右平移 i 位，利用切片防止越界
                    mask[:, i:] = mask[:, i:] | active_starts[:, :-i]

        x_mask = x[mask]
        y_mask = y[mask]
        p_mask = p[mask]
        if intensity is not None:
            intensity_mask = intensity[mask]
        tpe_mask = tpe[mask]
        
        # tokens = tokens.clone() # 只为了debug 平时可以去掉
        tokens[mask] = self.mask_token.to(tokens)



        if intensity is not None:
            if self.embed_fusion == 'plain':
                tokens = (tokens + tpe) * intensity.unsqueeze(2)
            elif self.embed_fusion == 'gated':
                tokens = self.embed_fusion_net(tokens, tpe, intensity.unsqueeze(2))
        else:
            if self.embed_fusion == 'plain':
                tokens = tokens + tpe
            elif self.embed_fusion == 'gated':
                tokens = self.embed_fusion_net(tokens, tpe)

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
        return {
            'n_predicts': n_mask.item(),
            'metrics': metrics,
            'loss': loss
        }

    def self_supervised_mae_forward(self, batch):
        p = batch['p']
        y = batch['y']
        x = batch['x']
        t = batch['t']
        if 'intensity' in batch:
            intensity = batch['intensity']
        else:
            intensity = None
        
        valid_mask = batch['valid_mask']

        # --- 1. 归一化 (保持不变) ---
        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.

        if intensity is not None:
            intensity = intensity.to(x)
            intensity = self.norm_intensity_fun(intensity)

        valid_mask_f = valid_mask.float()
        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]
        t = torch.cat((torch.zeros_like(t[:, 0:1]), t), dim=1)

        # --- 2. Embedding (保持不变) ---
        # [B, L, D]
        tokens = self.embed_net(torch.stack((x, y, p), dim=2), valid_mask)
        tpe = self.tpe(t) * valid_mask_f.unsqueeze(2)

        # --- 3. 生成 Mask (保持不变，你的几何 Mask 逻辑很好) ---
        B, L = valid_mask.shape
        if self.mask_len == 1:
            mask = torch.rand([B, L], device=t.device) < self.mask_ratio
        else:
            # (...这里省略掉你原有的几何分布 Mask 生成代码，逻辑保持完全一致...)
            # 假设这里已经生成了 mask 变量: mask[i, j] = True 表示被掩盖
            # 为了代码简洁，我直接复制你上面的 Mask 生成逻辑占位：
            prob = 1.0 / self.mask_len
            span_max_len = max(10, int(self.mask_len * 2))
            start_prob = self.mask_ratio / self.mask_len
            start_mask = torch.rand([B, L], device=t.device) < start_prob
            span_lens = torch.distributions.geometric.Geometric(probs=prob).sample([B, L]).to(t.device) + 1
            span_lens = span_lens.clamp(max=span_max_len).int()
            mask = torch.zeros([B, L], dtype=torch.bool, device=t.device)
            for i in range(span_max_len):
                active_starts = start_mask & (span_lens > i)
                if i == 0:
                    mask = mask | active_starts
                else:
                    mask[:, i:] = mask[:, i:] | active_starts[:, :-i]
            
        # 确保 mask 不会覆盖掉 padding 的区域 (虽然 padding 本身会被 valid_mask 处理，但双重保险)
        mask = mask & valid_mask

        # --- 4. 准备 Target (保持不变) ---
        x_mask = x[mask]
        y_mask = y[mask]
        p_mask = p[mask]
        # 注意：如果我们要预测 intensity，这里也需要保存 target
        if intensity is not None:
            intensity_mask = intensity[mask]
        tpe_mask = tpe[mask]

        # --- 5. MAE 核心逻辑：Encoder ---
        
        # 5.1 构造 Encoder 输入：只包含可见的 (Visible) Token
        # 注意：为了处理变长序列和 Batch 对齐，我们不物理删除 token，而是通过 Attention Mask 屏蔽
        
        encoder_input = tokens + tpe
        if intensity is not None:
            encoder_input = encoder_input * intensity.unsqueeze(2)
        
        # 关键点：构造 Encoder 的 Attention Mask
        # Encoder 只能看到 (有效事件 AND 未被Mask) 的位置
        encoder_padding_mask = valid_mask & (~mask) 
        
        # 5.2 Encoder 前向传播
        # self.backbone_net 需要支持传入 key_padding_mask 或 attention_mask
        # 输出的 encoded_features 在 mask 位置是无效的（因为没有看它们），在 unmask 位置是语义特征
        encoded_features = self.backbone_net(encoder_input, encoder_padding_mask)

        # --- 6. MAE 核心逻辑：Decoder / Reconstruction ---

        # 6.1 构造 Decoder 输入：
        # Visible 位置 -> 填入 Encoder 的输出
        # Masked 位置  -> 填入 Learnable Mask Token + Time Embedding
        
        decoder_tokens = encoded_features.clone()
        
        # 对被 Mask 的位置，填入 mask_token
        # 注意：MAE 中 Decoder 需要看到位置编码 (tpe) 来知道要预测哪里的事件
        # 且这里不能乘 intensity，因为 intensity 是我们要预测的内容（或者未知的）
        decoder_tokens[mask] = self.mask_token.to(tokens) + tpe[mask]
        
        # 如果有专门的 decoder_net (例如几层 Transformer)，在这里调用
        # 如果没有，可以直接用 backbone 的输出（但在 MAE 中通常建议加 1-2 层轻量级 Decoder）
        # Decoder 可以看到所有 token (包括 mask token 和 encoded visible tokens)
        
        decoder_output = self.decoder_net(self.enc_to_dec_proj(decoder_tokens), valid_mask)


        # --- 7. 预测与计算 Loss ---
        
        # 获取 Mask 位置的预测结果
        # 注意：BERT 逻辑里减去了 tpe_mask，这里看你具体 Head 的设计
        # 如果 Head 是预测 (Content + Pos) -> Content，则不需要减
        # 如果 Head 只是个 Linear，通常直接预测即可。
        # 照搬你原有的逻辑：
        if intensity is not None:
             # 原逻辑中有除以 intensity，但在 MAE 预测阶段 intensity 未知，
             # 所以我们直接预测 Raw Embedding 或者直接预测属性
             # 这里我们直接取 mask 对应的向量送入 Head
             predicted_features = decoder_output[mask]
        else:
             predicted_features = decoder_output[mask] - tpe_mask # 保持你原有的逻辑

        predicts = F.tanh(self.heads(predicted_features))
        
        # 假设输出通道顺序是 [p, y, x, (intensity)]
        p_predicted = predicts[..., 0]
        y_predicted = predicts[..., 1]
        x_predicted = predicts[..., 2]
        
        # 计算基础 Loss
        loss_p = F.mse_loss(p_predicted, p_mask, reduction='sum')
        loss_y = F.mse_loss(y_predicted, y_mask, reduction='sum')
        loss_x = F.mse_loss(x_predicted, x_mask, reduction='sum')
        
        total_loss = loss_p + loss_y + loss_x

        # --- 新增：预测 Intensity (如果存在) ---
        # 如果你的 self.heads 输出维度是 4，且包含 intensity
        if intensity is not None and predicts.shape[-1] > 3:
            rho_predicted = predicts[..., 3]
            # 对 intensity 这种长尾分布，通常建议预测 log(intensity) 或者使用专门的 Loss
            # 这里简单使用 MSE，注意 intensity_mask 应该也是归一化过的
            loss_rho = F.mse_loss(rho_predicted, intensity_mask, reduction='sum')
            total_loss += loss_rho

        n_mask = mask.long().sum()
        loss = total_loss / (n_mask + 1e-6) # 防止除零

        # --- 8. Metrics 计算 (保持不变) ---
        with torch.no_grad():
            true_y_pixel = ((y_mask + 1.) / 2. * (self.H - 1)).round().long()
            true_x_pixel = ((x_mask + 1.) / 2. * (self.W - 1)).round().long()
            true_p_final = ((p_mask + 1.) / 2.).round().long()

            metrics = utils.calculate_event_prediction_metrics(
                pred_p_logit=p_predicted,
                pred_y_norm=y_predicted,
                pred_x_norm=x_predicted,
                true_p=true_p_final,
                true_y=true_y_pixel,
                true_x=true_x_pixel,
                height=self.H,
                width=self.W,
                valid_mask=None,
                neighbors=3
            )
            
        return {
            'n_predicts': n_mask.item(),
            'metrics': metrics,
            'loss': loss
        }

    def classify_forward(self, batch):


        p = batch['p']
        y = batch['y']
        x = batch['x']
        t = batch['t']
        if 'intensity' in batch:
            intensity = batch['intensity']
        else:
            intensity = None
                          
        valid_mask = batch['valid_mask']

        # to [-1, 1]
        p *= (2. / (self.P - 1))
        p -= 1.
        y *= (2. / (self.H - 1))
        y -= 1.
        x *= (2. / (self.W - 1))
        x -= 1.

        if intensity is not None:
            intensity = self.norm_intensity_fun(intensity.to(x))



        t = t.float()
        t /= t.max(dim=1, keepdim=True)[0]
        valid_mask_f = valid_mask.float()
        t = torch.diff(t, dim=1) * valid_mask_f[:, 1:] * valid_mask_f[:, :-1]

        t = torch.cat((torch.zeros_like(t[:, 0:1]), t), dim=1)



        tokens = self.embed_net(torch.stack((x, y, p), dim=2), valid_mask)

        if intensity is not None:
            if self.training and self.p_intensity_drop > 0:
                mask = torch.rand_like(intensity) < self.p_intensity_drop
                intensity[mask] = 1

            tpe = self.tpe(t)
            

            if self.embed_fusion == 'plain':
                tokens = (tokens + tpe) * (intensity * valid_mask_f).unsqueeze(2)

            elif self.embed_fusion == 'gated':
                tokens = self.embed_fusion_net(tokens, tpe, (intensity * valid_mask_f).unsqueeze(2))



        else:
            if self.embed_fusion == 'plain':
                tokens = (tokens + self.tpe(t)) * valid_mask_f.unsqueeze(2)

            elif self.embed_fusion == 'gated':
                tokens = self.embed_fusion_net(tokens, self.tpe(t) * valid_mask_f.unsqueeze(2))

        if self.training and self.p_token_mix > 0:
            if torch.rand(size=[1]).item() < self.p_token_mix:
                tokens, batch['label'] = self.mixer(tokens, batch['label'])


        if self.pool_every_layer > 0:
            for i in range(len(self.backbone_net)):
                tokens = self.backbone_net[i](tokens, valid_mask)
                tokens, valid_mask = sequence_avg_pooling(tokens=tokens, valid_mask=valid_mask, stride=2)
        else:
            tokens = self.backbone_net(tokens, valid_mask)

        valid_mask_f = valid_mask.float()
       
        tokens = (tokens * valid_mask_f.unsqueeze(2)).sum(1) / (valid_mask_f.sum(dim=1, keepdim=True) + 1e-5)
        tokens = self.heads(tokens)

        if self.contrastive_learning_temperature > 0:
            tokens = F.normalize(tokens, dim=1)
            loss = self.contrastive_learning_loss(tokens, batch['label'])
            return {'loss': loss}
        else:
            return {'predicts': tokens}
