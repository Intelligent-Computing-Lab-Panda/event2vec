import lightning
import torch
import numpy as np
from . import utils

from .asl_dvs import ASLDVS
from .dvs_gesture import DVSGesture
from .dvs_lip import DVSLip

class EventDataModule(lightning.LightningDataModule):
    def __init__(self, name: str, root: str, sample_number: int, sampler: str, batch_size: int,
                 num_workers: int,
                 train_repeats:int=1,
                 val_root:str=None, val_sample_number:int=None, val_sampler:str=None,
                 val_repeats:int=8
                 ):
        super().__init__()
        self.name = name
        self.root = root
        self.sample_number = sample_number

        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_repeats = train_repeats

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







    def setup(self, stage: str):
        if self.name == 'dvs_gesture':
            dts_class = DVSGesture
        elif self.name == 'asl_dvs':
            dts_class = ASLDVS
        elif self.name == 'dvs_lip':
            dts_class = DVSLip




        else:
            raise NotImplementedError(self.name)

        self.train_set = dts_class(training=True, root=self.root, sample_number=self.sample_number,
                                    sampler=self.sampler, repeats=self.train_repeats)

        self.val_set = dts_class(training=False, root=self.val_root, sample_number=self.val_sample_number,
                                  sampler=self.val_sampler, repeats=self.val_repeats)


    def train_dataloader(self):


        collate_fn = utils.event_collate_fun_with_padding
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, drop_last=True,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=utils.seed_worker,
                                           persistent_workers=self.num_workers > 0,
                                           prefetch_factor=4 if self.num_workers > 0 else None)

    def val_dataloader(self):

        collate_fn = utils.event_collate_fun_with_padding

        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, drop_last=False,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=utils.seed_worker,
                                           persistent_workers=self.num_workers > 0,
                                           prefetch_factor=4 if self.num_workers > 0 else None)




if __name__ == '__main__':

    pass







