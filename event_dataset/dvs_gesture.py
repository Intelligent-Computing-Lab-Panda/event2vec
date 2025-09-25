import os
import types

from .utils import EventNPDataset

class DVSGesture(EventNPDataset):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, repeats:int):
        if training:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')
        super().__init__(training=training, root=root, sample_number=sample_number, sampler=sampler, repeats=repeats)


    @staticmethod
    def num_classes():
        return 11

    @staticmethod
    def event_size():
        P = 2
        H = 128
        W = 128
        return P, H, W
