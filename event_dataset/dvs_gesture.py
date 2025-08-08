import os
import types

from .utils import EventNPDataset

class DVSGesture(EventNPDataset):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, norm_t: str, full_sample: bool, return_index:bool, repeats:int):
        if training:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')
        super().__init__(training=training, root=root, sample_number=sample_number, sampler=sampler, norm_t=norm_t, full_sample=full_sample, return_index=return_index, repeats=repeats)

        self.t_max = 18456845

    @staticmethod
    def num_classes():
        return 11

    @staticmethod
    def event_size():
        P = 2
        H = 128
        W = 128
        return P, H, W

    def statistics(self):
        if self.training:
            t_min, t_max, n_min, n_max, n_events = 1749829, 18456845, 35267, 1594557, 425993547
        else:
            t_min, t_max, n_min, n_max, n_events = 1798339, 15591318, 73010, 1459889, 118206899

        return {'t_min': t_min, 't_max': t_max, 'n_min': n_min, 'n_max': n_max, 'n_events': n_events}