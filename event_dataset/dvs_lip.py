import os
import types
from .utils import EventNPDataset

class DVSLip(EventNPDataset):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, repeats:int=1):
        if training:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        super().__init__(training=training, root=root, sample_number=sample_number, sampler=sampler, repeats=repeats)

    @staticmethod
    def num_classes():
        return 100

    @staticmethod
    def event_size():
        P = 2
        H = 128
        W = 128
        return P, H, W

import os
import numpy as np
from .utils import k_mean_cluster
def cluster_dvs_lip(out_dir, n_clusters=1024, root = '/dev/shm/dvs_lip', save:bool=True):
    # this is an example code to cluster dvs_lip by k-means
    # this function is not optimal by multi-processing and faiss (GPU)
    for train in (True, False):
        dts = DVSLip(training=train, root=root, sample_number=0, sampler=None, repeats=1)
        idx_to_class = {v: k for k, v in dts.class_to_idx.items()}

        for i in range(len(dts)):
            x, y, t, p, intensity, label = dts[i]
            label = idx_to_class[label]
            path = dts.samples[i][0]
            
            x, y, t, p, intensity = k_mean_cluster(x.numpy(), y.numpy(), t.numpy(), p.numpy(), n_clusters, H=128, W=128, scale_t=1)
            fname = os.path.basename(path)
            label_dir = os.path.join(out_dir, 'train' if train else 'test', label)
            fname = os.path.join(label_dir, fname)
            if save:
                os.makedirs(label_dir, exist_ok=True)
                np.savez(fname, x=x, y=y, t=t, p=p, intensity=intensity)
                print('Save to', fname)
            else:
                # debug
                print('if save=True, will save to', fname)