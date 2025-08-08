import lightning
import torch
import numpy as np
from . import utils

from .asl_dvs import ASLDVS
from .dvs_gesture import DVSGesture

class EventDataModule(lightning.LightningDataModule):
    def __init__(self, name: str, root: str, sample_number: int, sampler: str, norm_t: str, batch_size: int,
                 num_workers: int,
                 train_full_sample: bool,
                 val_root:str=None, val_sample_number:int=None, val_sampler:str=None,
                 val_repeats:int=8
                 ):
        super().__init__()
        self.name = name
        self.root = root
        self.sample_number = sample_number

        self.sampler = sampler
        self.norm_t = norm_t
        self.batch_size = batch_size
        self.num_workers = num_workers

        if val_root is None:
            val_root = root
        if val_sample_number is None:
            val_sample_number = sample_number
        if val_sampler is None:
            val_sampler = sampler
        self.val_root = val_root
        self.val_sample_number = val_sample_number
        self.val_sampler = val_sampler
        self.val_repeats = val_repeats
        self.train_full_sample = train_full_sample






    def setup(self, stage: str):
        if self.name == 'dvs_gesture':
            dts_class = DVSGesture
        elif self.name == 'asl_dvs':
            dts_class = ASLDVS

        else:
            raise NotImplementedError(self.name)

        self.train_set = dts_class(training=True, root=self.root, sample_number=self.sample_number,
                                    sampler=self.sampler, norm_t=self.norm_t, full_sample=self.train_full_sample, return_index=False, repeats=1)

        self.val_set = dts_class(training=False, root=self.val_root, sample_number=self.val_sample_number,
                                  sampler=self.val_sampler, norm_t=self.norm_t, full_sample=False, return_index=False, repeats=self.val_repeats)


    def train_dataloader(self):


        collate_fn = utils.event_collate_fun_with_padding
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, drop_last=True,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=utils.seed_worker,
                                           persistent_workers=True,
                                           prefetch_factor=4)

    def val_dataloader(self):
        if self.val_set.return_index:
            collate_fn = utils.event_collate_fun_with_padding_and_index
        else:
            collate_fn = utils.event_collate_fun_with_padding

        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, drop_last=False,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=utils.seed_worker,
                                           persistent_workers=True,
                                           prefetch_factor=4)




if __name__ == '__main__':
    pass





