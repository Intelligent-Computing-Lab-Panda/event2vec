import datetime
import math
import time
from collections import OrderedDict

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
from torchmetrics import MeanMetric, Precision
from torchmetrics.classification import Accuracy
import event_transform, event_embedding
import model_zoo



def configure_param_lr_wd(m: nn.Module, lr: float, wd: float, encoder_lr_decay_rate: float=0.75, deacy_lr_encoder_layers:nn.Sequential=None):
    p_flag = OrderedDict()
    for p in m.parameters():
        if p.requires_grad:
            key = id(p)
            assert key not in p_flag
            p_flag[key] = {'params': p, 'weight_decay': wd, 'lr': lr}
    weight_decay_modules = m
    if hasattr(m, 'weight_decay_modules'):
        weight_decay_modules = m.weight_decay_modules()

    for module_name, module in weight_decay_modules.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{module_name}.{param_name}" if module_name else param_name

            # check if it needs weight_decay
            if wd > 0:
                if param_name in ('bias', 'class_token', 'mask_token', 'pos_embedding', 'pe_class_token') or \
                        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.RMSNorm, nn.Embedding, )):
                    p_flag[id(param)]['weight_decay'] = 0.

                elif 'token' in param_name:
                    print(f'{full_param_name} is regarded as with weight decay.')

    if deacy_lr_encoder_layers is not None:
        depth = len(deacy_lr_encoder_layers)
        for i in range(depth):
            dlr = lr * (encoder_lr_decay_rate ** (depth - i))
            for p in deacy_lr_encoder_layers[i].parameters():
                if p.requires_grad:
                    p_flag[id(p)]['lr'] = dlr



    return p_flag.values()



class Event2VecClassifier(lightning.LightningModule):
    def __init__(self, P: int, H: int, W: int, h: int, w: int,

                 train_transform_args: str,
                 train_transform_policy: str,
                 test_transform_args: str,

                 backbone:str=None,

                 n_classes: int = -1,

                 compile: bool = False,
                 lr: float = 1e-3,
                 min_lr: float=0., # 1e-6 for finetune
                 batch_size: int=-1,
                 warmup_epochs:int=0,
                optimizer: str = 'adamw',
                 lrs:str='CosineAnnealingLR',
                 wd: float = 0.,
                 label_smoothing: float = 0.,
                 load:str=None,
                 ):
        super().__init__()

        self.P = P
        self.H = H
        self.W = W
        self.h = h
        self.w = w

        self.transforms = event_transform.get_transform_module(train_transform_args=train_transform_args,
                                                               train_transform_policy=train_transform_policy,
                                                               test_transform_args=test_transform_args, H=H, W=W, h=h, w=w)
        # event transforms, just like torchvision.transforms



        self.classifier = model_zoo.create_classifier(backbone=backbone, n_classes=n_classes)



        self.compile = compile
        self.lr = lr
        self.min_lr = min_lr
        self.lrs = lrs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.optimizer_name = optimizer.lower()
        self.wd = wd
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes


        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=n_classes)







        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

        self.compile_flag = compile
        self.print_info = ''
        self.train_duration = 0.
        self.valid_duration = 0.

        self.load = load
        if load:
            incompatible_keys = self.load_state_dict(torch.load(load, map_location='cpu')['state_dict'], strict=False)

            if self.global_rank == 0:
                if incompatible_keys.missing_keys:
                    print('missing state dict keys:\n', incompatible_keys)


        n_param = 0
        for p in self.parameters():
            if p.requires_grad:
                n_param += p.numel()

        print('param in MB:', n_param * 4 / 1024 / 1024)
        time.sleep(1)

    def training_step(self, batch, batch_idx):
        xytp, valid_mask, label = batch
        self.train_samples += label.shape[0]
        outputs = self(xytp, valid_mask)



        label_predicted = outputs

        loss = F.cross_entropy(label_predicted, label, label_smoothing=self.label_smoothing)

        self.train_acc.update(label_predicted, label)


        self.train_loss.update(loss.data)
        return loss

    def validation_step(self, batch, batch_idx):
        xytp, valid_mask, label = batch



        self.val_samples += label.shape[0]

        outputs = self(xytp, valid_mask)


        label_predicted = outputs
        loss = F.cross_entropy(label_predicted, label, label_smoothing=self.label_smoothing)

        self.valid_acc.update(label_predicted, label)




        self.valid_loss.update(loss.data)
        return loss


    def on_validation_epoch_start(self):
        self.val_samples = 0
        self.valid_start_time = time.time()


    def on_validation_epoch_end(self):

        valid_acc = self.valid_acc.compute()
        self.valid_acc.reset()
        self.log('valid_acc', valid_acc, on_epoch=True)


        valid_loss = self.valid_loss.compute()

        self.log('valid_loss', valid_loss, on_epoch=True)

        self.valid_loss.reset()
        self.valid_end_time = time.time()
        self.valid_duration = self.valid_end_time - self.valid_start_time
        self.valid_speed = self.val_samples / self.valid_duration * self.trainer.world_size
        self.val_samples = 0

        if self.global_rank == 0:
            print(
                f'valid_loss={valid_loss:.3f}, valid_acc={valid_acc:.3f}, valid_speed={self.valid_speed:.3f} samples/sec')


            print(
                f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(self.train_duration + self.valid_duration) * (self.trainer.max_epochs - self.current_epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


    def on_train_epoch_start(self):
        self.train_samples = 0
        self.train_start_time = time.time()

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        self.train_acc.reset()


        train_loss = self.train_loss.compute()
        if self.global_rank == 0:
            print(self.print_info)

        self.log('train_acc', train_acc, on_epoch=True)


        self.log('train_loss', train_loss, on_epoch=True)

        self.train_loss.reset()
        self.train_end_time = time.time()
        self.train_duration = self.train_end_time - self.train_start_time
        self.train_speed = self.train_samples / self.train_duration * self.trainer.world_size
        self.train_samples = 0
        if self.global_rank == 0:
            print(
                f'epoch={self.current_epoch}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}, train_speed={self.train_speed:.3f} samples/sec')



    def forward(self, xytp, valid_mask):
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]

        p, y, x, t, valid_mask = self.transforms(p, y, x, t, valid_mask)
        '''
        event transform (data augmentation)
        '''



        t = t - t[:, 0].unsqueeze(1)
        '''
        let t start from 0
        '''
        return self.classifier(p, y, x, t, valid_mask)













    def configure_optimizers(self):
        lr = self.lr * self.batch_size * self.trainer.world_size / 256
        # if self.load is None:
        #     encoder_lr_decay_rate = -1
        #     deacy_lr_encoder_layers = None
        # else:
        #     encoder_lr_decay_rate = 0.75
        #     deacy_lr_encoder_layers = self.classifier.deacy_lr_encoder_layers

        encoder_lr_decay_rate = -1
        deacy_lr_encoder_layers = None
        param_groups = configure_param_lr_wd(self.classifier, lr, self.wd, encoder_lr_decay_rate, deacy_lr_encoder_layers)
        # check
        assert len(list(self.parameters())) == len(list(self.classifier.parameters()))

        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(param_groups, fused=True)
        else:
            raise NotImplementedError(self.optimizer_name)

        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs)
        else:
            warmup_scheduler = None

        if self.lrs.startswith('CosineAnnealingLR'):
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs - self.warmup_epochs ,eta_min=self.min_lr)
        elif self.lrs.startswith('CosineAnnealingWarmRestarts-'):
            # CosineAnnealingWarmRestarts-{T_0,T_mult}
            temp = self.lrs.split('-')[1].split(',')
            T_0 = int(temp[0])
            T_mult = int(temp[1])
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=T_0, T_mult=T_mult,
                                                                        eta_min=self.min_lr, last_epoch=self.trainer.max_epochs)
        elif self.lrs.startswith('StepLR-'):
            # StepLR-{step_size}-{gamma}
            temp = self.lrs.split('-')
            step_size = int(temp[1])
            gamma = float(temp[2])
            main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=self.trainer.max_epochs)
        elif self.lrs.lower() == 'none':
            main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            raise NotImplementedError(self.lrs)

        if warmup_scheduler is not None:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs]
            )
        else:
            lr_scheduler = main_scheduler


        if lr_scheduler is None:
            return optimizer
        else:
            return ([optimizer], [lr_scheduler])