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
                 attention: str, d_model: int, d_feedforward: int, nheads: int, n_layers: int,
                 n_classes: int, activation: str, mask_ratio:float, mask_len:int, p_token_mix:float, p_intensity_drop:float, drop_path:float, pool_every_layer:int,
                 bi_share_param:bool, intensity_norm:str, train_head_only:bool, remove_load_keys:str,
                 contrastive_learning_temperature:float,
                 self_supervised_style:str, spatial_embed:str, temporal_embed:str, embed_fusion:str,

                train_transform_args: str,
                train_transform_policy: str,
                test_transform_args: str,
                compile_flag: bool = False,
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



        self.classifier = custom.E2VNet(P=P, H=H, W=W, attention=attention, d_model=d_model, d_feedforward=d_feedforward, nheads=nheads, n_layers=n_layers, n_classes=n_classes, activation=activation, mask_ratio=mask_ratio, mask_len=mask_len, p_token_mix=p_token_mix, p_intensity_drop=p_intensity_drop, drop_path=drop_path, pool_every_layer=pool_every_layer, bi_share_param=bi_share_param, intensity_norm=intensity_norm, contrastive_learning_temperature=contrastive_learning_temperature, self_supervised_style=self_supervised_style, spatial_embed=spatial_embed, temporal_embed=temporal_embed, embed_fusion=embed_fusion)


        self.lr = lr
        self.min_lr = min_lr
        self.lrs = lrs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.optimizer_name = optimizer.lower()
        self.wd = wd
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        self.train_head_only = train_head_only

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

        self.compile_flag = compile_flag
        self.print_info = ''
        self.train_duration = 0.
        self.valid_duration = 0.

        self.load = load
        if load:
            state_dict = torch.load(load, map_location='cpu')['state_dict']

            # 移除由于compile导致的模型参数前缀 _orig_mod
            new_state_dict = {}
            for key, value in state_dict.items():
                # 如果 key 包含这个前缀，则替换掉
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value

            state_dict = new_state_dict
            keys_to_be_removed = []
            for key in state_dict.keys():
                if key.startswith(remove_load_keys):
                    keys_to_be_removed.append(key)
            
            for key in keys_to_be_removed:
                del state_dict[key]

            incompatible_keys = self.load_state_dict(state_dict, strict=False)

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

    def setup(self, stage):
        if self.compile_flag:
            self.classifier = torch.compile(self.classifier)
        return super().setup(stage)

    def training_step(self, batch, batch_idx):
        self.train_samples += batch['label'].shape[0]
        outputs = self(batch)


        if self.classifier.self_supervised_training:
            loss = outputs['loss']
            for key, value in outputs['metrics'].items():
                self.train_acc[key].update(value, weight=outputs['n_predicts'])
        else:
            if self.classifier.contrastive_learning_temperature > 0:
                loss = outputs['loss']
            else:
                loss = F.cross_entropy(outputs['predicts'], batch['label'], label_smoothing=self.label_smoothing)
                if batch['label'].dim() == 2:
                    batch['label'] = batch['label'].argmax(1)
                self.train_acc.update(outputs['predicts'], batch['label'])


        self.train_loss.update(loss.data)
        return loss

    def validation_step(self, batch, batch_idx):
       
        self.val_samples += batch['label'].shape[0]

        outputs = self(batch)

        if self.classifier.self_supervised_training:
            loss = outputs['loss']
            for key, value in outputs['metrics'].items():
                self.valid_acc[key].update(value, weight=outputs['n_predicts'])
        else:
            if self.classifier.contrastive_learning_temperature > 0:
                loss = outputs['loss']
            else:
                loss = F.cross_entropy(outputs['predicts'], batch['label'], label_smoothing=self.label_smoothing)

                if batch['label'].dim() == 2:
                    batch['label'] = batch['label'].argmax(1)
                self.valid_acc.update(outputs['predicts'], batch['label'])

                if self.trainer.datamodule.val_set.repeats > 1:
                    repeat_indices = batch['indices'] // self.n_unique_samples
                    correct_mask = (outputs['predicts'].argmax(1) == batch['label'])
                    batch_total_counts = torch.bincount(repeat_indices, minlength=self.trainer.datamodule.val_set.repeats)
                    correct_repeat_indices = repeat_indices[correct_mask]
                    batch_correct_counts = torch.bincount(correct_repeat_indices, minlength=self.trainer.datamodule.val_set.repeats)

                    self.correct_counts_per_repeat += batch_correct_counts
                    self.total_counts_per_repeat += batch_total_counts

        self.valid_loss.update(loss.data)
        return loss



    def on_validation_epoch_start(self):
        self.val_samples = 0
        self.valid_start_time = time.time()

        if self.trainer.datamodule.val_set.repeats > 1 and self.classifier.contrastive_learning_temperature <= 0:
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
            if self.classifier.contrastive_learning_temperature > 0:
                valid_acc = 0.
                valid_acc_std = 0.
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
            if self.classifier.contrastive_learning_temperature > 0:
                train_acc = 0.
                
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
            print(self.trainer.log_dir, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if self.classifier.self_supervised_training:
                print(
                    f'epoch={self.current_epoch}, train_loss={train_loss:.6f}, train_speed={self.train_speed:.6f} samples/sec')
                for key, value in train_acc.items():
                    print(f'{key}={value: .6f}', end=', ')
                print('\n')
            else:
                print(
                    f'epoch={self.current_epoch}, train_loss={train_loss:.6f}, train_acc={train_acc:.6f}, train_speed={self.train_speed:.6f} samples/sec')



    def forward(self, batch):

        p, y, x, t, valid_mask, label = self.transforms(batch['p'].clone(), batch['y'].clone(), batch['x'].clone(), batch['t'].clone(), batch['valid_mask'].clone(), batch['label'].clone())

        if self.classifier.contrastive_learning_temperature > 0:
            p_, y_, x_, t_, valid_mask_, label_ = self.transforms(batch['p'], batch['y'], batch['x'], batch['t'], batch['valid_mask'], batch['label'])
            p = torch.cat((p, p_), 0)
            y = torch.cat((y, y_), 0)
            x = torch.cat((x, x_), 0)
            t = torch.cat((t, t_), 0)
            valid_mask = torch.cat((valid_mask, valid_mask_), 0)
            label = torch.cat((label, label_), 0)
            if 'intensity' in batch.keys():
                batch['intensity'] = torch.cat((batch['intensity'], batch['intensity']), 0)


        batch['p'], batch['y'], batch['x'], batch['t'], batch['valid_mask'], batch['label'] = p, y, x, t, valid_mask, label
        '''
        event transform (data augmentation)
        '''
        batch['t'] = batch['t'] - batch['t'][:, 0].unsqueeze(1)
        '''
        let t start from 0
        '''
        return self.classifier(batch)













    def configure_optimizers(self):
        lr = self.lr * self.batch_size * self.trainer.world_size / 256
        encoder_lr_decay_rate = -1
        deacy_lr_encoder_layers = None
        if self.train_head_only:
            for p in self.classifier.parameters():
                p.requires_grad = False
            for p in self.classifier.heads.parameters():
                p.requires_grad = True
            
        param_groups = utils.configure_param_lr_wd(self.classifier, lr, self.wd, encoder_lr_decay_rate, deacy_lr_encoder_layers)
        if not self.train_head_only:
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
