import datetime
import time


import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy
import event_transform, utils
from model_zoo import custom


class Event2VecClassifier(lightning.LightningModule):
    def __init__(self, P: int, H: int, W: int, h: int, w: int,
                 d_model: int, d_feedforward: int, nheads: int, n_layers: int,
                 n_classes: int, activation: str, mask_ratio:float, p_token_mix:float, p_intensity_drop:float, drop_path:float, pool_every_layer:int,

                train_transform_args: str,
                train_transform_policy: str,
                test_transform_args: str,
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



        self.classifier = custom.E2VNet(P=P, H=H, W=W, d_model=d_model, d_feedforward=d_feedforward, nheads=nheads, n_layers=n_layers, n_classes=n_classes, activation=activation, mask_ratio=mask_ratio, p_token_mix=p_token_mix, p_intensity_drop=p_intensity_drop, drop_path=drop_path, pool_every_layer=pool_every_layer)



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

        if self.classifier.self_supervised_training:

            self.train_acc = nn.ModuleDict({
                'p_accuracy': MeanMetric(),
                'mae_y': MeanMetric(),
                'mae_x': MeanMetric(),
                'exact_match_accuracy': MeanMetric(),
                'neighbor_accuracy': MeanMetric()})
            self.valid_acc = nn.ModuleDict({
                'p_accuracy': MeanMetric(),
                'mae_y': MeanMetric(),
                'mae_x': MeanMetric(),
                'exact_match_accuracy': MeanMetric(),
                'neighbor_accuracy': MeanMetric()})

            self.test_acc = nn.ModuleDict({
                'p_accuracy': MeanMetric(),
                'mae_y': MeanMetric(),
                'mae_x': MeanMetric(),
                'exact_match_accuracy': MeanMetric(),
                'neighbor_accuracy': MeanMetric()})

        else:
            self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
            self.valid_acc = Accuracy(task="multiclass", num_classes=n_classes)
            self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)







        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.test_loss = MeanMetric()

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
                else:
                    print('all keys are loaded')

        n_p = 0
        for p in self.parameters():
            if p.requires_grad:
                n_p += p.numel()
        print('params in MB', n_p * 4 / 1024 / 1024)


    def training_step(self, batch, batch_idx):
        xytp, intensity, valid_mask, target = batch
        self.train_samples += target.shape[0]
        outputs, target = self(xytp, intensity, valid_mask, target)


        if self.classifier.self_supervised_training:
            n_predicts, metrics, loss = outputs
            for key, value in metrics.items():
                self.train_acc[key].update(value, weight=n_predicts)
        else:
            label_predicted = outputs

            loss = F.cross_entropy(label_predicted, target, label_smoothing=self.label_smoothing)
            if target.dim() == 2:
                target = target.argmax(1)
            self.train_acc.update(label_predicted, target)


        self.train_loss.update(loss.data)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.datamodule.val_set.repeats > 1:
            xytp, intensity, valid_mask, target, indices = batch
        else:
            xytp, intensity, valid_mask, target = batch



        self.val_samples += target.shape[0]

        outputs, target = self(xytp, intensity, valid_mask, target)

        if self.classifier.self_supervised_training:
            n_predicts, metrics, loss = outputs
            for key, value in metrics.items():
                self.valid_acc[key].update(value, weight=n_predicts)
        else:
            label_predicted = outputs
            loss = F.cross_entropy(label_predicted, target, label_smoothing=self.label_smoothing)

            if target.dim() == 2:
                target = target.argmax(1)
            self.valid_acc.update(label_predicted, target)

            if self.trainer.datamodule.val_set.repeats > 1:
                repeat_indices = indices // self.n_unique_samples
                correct_mask = (label_predicted.argmax(1) == target)
                batch_total_counts = torch.bincount(repeat_indices, minlength=self.trainer.datamodule.val_set.repeats)
                correct_repeat_indices = repeat_indices[correct_mask]
                batch_correct_counts = torch.bincount(correct_repeat_indices, minlength=self.trainer.datamodule.val_set.repeats)

                self.correct_counts_per_repeat += batch_correct_counts
                self.total_counts_per_repeat += batch_total_counts

        self.valid_loss.update(loss.data)
        return loss

    def test_step(self, batch, batch_idx):
        if self.trainer.datamodule.test_set.repeats > 1:
            xytp, intensity, valid_mask, target, indices = batch
        else:
            xytp, intensity, valid_mask, target = batch



        self.test_samples += target.shape[0]

        outputs, target = self(xytp, intensity, valid_mask, target)

        if self.classifier.self_supervised_training:
            n_predicts, metrics, loss = outputs
            for key, value in metrics.items():
                self.test_acc[key].update(value, weight=n_predicts)
        else:
            label_predicted = outputs
            loss = F.cross_entropy(label_predicted, target, label_smoothing=self.label_smoothing)

            if target.dim() == 2:
                target = target.argmax(1)
            self.test_acc.update(label_predicted, target)

            if self.trainer.datamodule.test_set.repeats > 1:
                repeat_indices = indices // self.n_unique_samples
                correct_mask = (label_predicted.argmax(1) == target)
                batch_total_counts = torch.bincount(repeat_indices, minlength=self.trainer.datamodule.test_set.repeats)
                correct_repeat_indices = repeat_indices[correct_mask]
                batch_correct_counts = torch.bincount(correct_repeat_indices, minlength=self.trainer.datamodule.test_set.repeats)

                self.correct_counts_per_repeat += batch_correct_counts
                self.total_counts_per_repeat += batch_total_counts

        self.test_loss.update(loss.data)
        return loss

  

    def on_validation_epoch_start(self):
        self.val_samples = 0
        self.valid_start_time = time.time()

        if self.trainer.datamodule.val_set.repeats > 1:
            self.n_unique_samples = len(self.trainer.datamodule.val_set) // self.trainer.datamodule.val_set.repeats
            self.correct_counts_per_repeat = torch.zeros(self.trainer.datamodule.val_set.repeats, device=self.device)
            self.total_counts_per_repeat = torch.zeros(self.trainer.datamodule.val_set.repeats, device=self.device)

    def on_validation_epoch_end(self):

        if self.classifier.self_supervised_training:
            valid_acc = {}
            for key in self.valid_acc.keys():
                value = self.valid_acc[key].compute()
                valid_acc[key] = value
                self.valid_acc[key].reset()
                self.log('val_' + key, value, on_epoch=True)

        else:
            valid_acc = self.valid_acc.compute()
            self.valid_acc.reset()
            self.log('valid_acc', valid_acc, on_epoch=True)
            valid_acc_std = 0.
            if self.trainer.datamodule.val_set.repeats > 1:
                self.all_gather(self.correct_counts_per_repeat).sum(0)
                self.all_gather(self.total_counts_per_repeat).sum(0)
                accuracies_per_repeat = self.correct_counts_per_repeat / self.total_counts_per_repeat

                valid_acc_std = accuracies_per_repeat.std()
                self.log('valid_acc_std', valid_acc_std, on_epoch=True)

        valid_loss = self.valid_loss.compute()

        self.log('valid_loss', valid_loss, on_epoch=True)

        self.valid_loss.reset()
        self.valid_end_time = time.time()
        self.valid_duration = self.valid_end_time - self.valid_start_time
        self.valid_speed = self.val_samples / self.valid_duration * self.trainer.world_size
        self.val_samples = 0

        if self.global_rank == 0:
            if self.classifier.self_supervised_training:
                print(
                    f'valid_loss={valid_loss:.6f}, valid_speed={self.valid_speed:.6f} samples/sec', end=', ')
                for key, value in valid_acc.items():
                    print(f'{key}={value: .6f}', end=', ')
                print('\n')
            else:
                print(
                    f'valid_loss={valid_loss:.6f}, valid_acc={valid_acc:.6f}, valid_acc_std={valid_acc_std: .6f}, valid_speed={self.valid_speed:.6f} samples/sec')


            print(
                f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(self.train_duration + self.valid_duration) * (self.trainer.max_epochs - self.current_epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')




    def on_train_epoch_start(self):
        self.train_samples = 0
        self.train_start_time = time.time()

    def on_train_epoch_end(self):


        if self.classifier.self_supervised_training:
            # start_ratio = 0.05
            # end_ratio = 0.4
            # current_ratio = start_ratio + (end_ratio - start_ratio) * (
            #             self.trainer.current_epoch / self.trainer.max_epochs)
            # self.classifier.mask_ratio = current_ratio
            # if self.global_rank == 0:
            #     print('set mask ratio =', current_ratio)

            train_acc = {}
            for key in self.train_acc.keys():
                value = self.train_acc[key].compute()
                train_acc[key] = value
                self.train_acc[key].reset()
                self.log('train_' + key, value, on_epoch=True)

        else:
            train_acc = self.train_acc.compute()
            self.train_acc.reset()
            self.log('train_acc', train_acc, on_epoch=True)

        train_loss = self.train_loss.compute()
        if self.global_rank == 0:
            print(self.print_info)



        self.log('train_loss', train_loss, on_epoch=True)

        self.train_loss.reset()
        self.train_end_time = time.time()
        self.train_duration = self.train_end_time - self.train_start_time
        self.train_speed = self.train_samples / self.train_duration * self.trainer.world_size
        self.train_samples = 0
        if self.global_rank == 0:
            if self.classifier.self_supervised_training:
                print(
                    f'epoch={self.current_epoch}, train_loss={train_loss:.6f}, train_speed={self.train_speed:.6f} samples/sec')
                for key, value in train_acc.items():
                    print(f'{key}={value: .6f}', end=', ')
                print('\n')
            else:
                print(
                    f'epoch={self.current_epoch}, train_loss={train_loss:.6f}, train_acc={train_acc:.6f}, train_speed={self.train_speed:.6f} samples/sec')



    def forward(self, xytp, intensity, valid_mask, target):
        x = xytp[0]
        y = xytp[1]
        t = xytp[2]
        p = xytp[3]

        if self.training and self.trainer.datamodule.name == 'n_cars':
            # 针对n_cars额外增加的
            # 1. 保存原始样本的副本以备回退
            p_orig, y_orig, x_orig, t_orig, valid_mask_orig = p.clone(), y.clone(), x.clone(), t.clone(), valid_mask.clone()

            # 2. 应用数据增强
            p_aug, y_aug, x_aug, t_aug, valid_mask_aug, target = self.transforms(p, y, x, t, valid_mask, target)

            # 3. 计算每个样本增强后的有效事件数量
            # valid_mask_aug 的形状是 [B, L], .sum(dim=1) 后得到形状为 [B] 的张量
            valid_lengths = valid_mask_aug.sum(dim=1)

            # 4. 找出需要恢复的样本
            # revert_mask 的形状是 [B], True 表示该样本需要被恢复
            revert_mask = valid_lengths < 16

            # 5. 如果有任何样本需要恢复，就执行替换操作
            if torch.any(revert_mask):
                # 为了使用 torch.where，需要将 revert_mask 的维度从 [B] 扩展到 [B, 1]，
                # 以便与形状为 [B, L] 的张量进行广播操作
                revert_mask_expanded = revert_mask.unsqueeze(1)

                # 使用 torch.where 根据 revert_mask_expanded 的值，
                # 从原始张量 (p_orig, ...) 或增强后的张量 (p_aug, ...) 中选择数据
                p = torch.where(revert_mask_expanded, p_orig, p_aug)
                y = torch.where(revert_mask_expanded, y_orig, y_aug)
                x = torch.where(revert_mask_expanded, x_orig, x_aug)
                t = torch.where(revert_mask_expanded, t_orig, t_aug)
                valid_mask = torch.where(revert_mask_expanded, valid_mask_orig, valid_mask_aug)
            else:
                # 如果没有样本需要恢复，则直接使用增强后的结果
                p, y, x, t, valid_mask = p_aug, y_aug, x_aug, t_aug, valid_mask_aug
        else:
            p, y, x, t, valid_mask, target = self.transforms(p, y, x, t, valid_mask, target)
        '''
        event transform (data augmentation)
        '''
        t = t - t[:, 0].unsqueeze(1)
        '''
        let t start from 0
        '''
        return self.classifier(p, y, x, t, intensity, valid_mask, target)













    def configure_optimizers(self):
        lr = self.lr * self.batch_size * self.trainer.world_size / 256
        encoder_lr_decay_rate = -1
        deacy_lr_encoder_layers = None
        param_groups = utils.configure_param_lr_wd(self.classifier, lr, self.wd, encoder_lr_decay_rate, deacy_lr_encoder_layers)
        # check
        assert len(list(self.parameters())) == len(list(self.classifier.parameters()))

        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(param_groups, fused=self.trainer.gradient_clip_algorithm is None)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, fused=self.trainer.gradient_clip_algorithm is None)
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