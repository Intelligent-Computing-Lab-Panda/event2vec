import math
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from sympy.combinatorics.group_numbers import groups_count
from torch import dtype
from torch.nn.utils.rnn import pad_sequence
import lightning
import h5py
import numba
import importlib
import heapq
from sklearn.cluster import KMeans

from concurrent.futures import ThreadPoolExecutor



def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.initial_seed()
    worker_rng = np.random.default_rng(worker_seed)
    worker_info.dataset.rng = worker_rng


def random_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                  sample_number: int):
    n = x.size
    if n > sample_number:
        indices = rng.choice(n, sample_number, replace=False)
        indices.sort()
        return x[indices], y[indices], t[indices], p[indices]
    else:
        return x, y, t, p


def k_mean_cluster(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                  sample_number: int, H: int, W: int, scale_t: float):
    n = x.size
    if n <= sample_number:
        return x, y, t, p, np.ones_like(x)

    t -= t[0]
    points = np.column_stack((
        x.astype(float) / (W - 1),
        y.astype(float) / (H - 1),
        t.astype(float) / t[-1] * scale_t
    ))
    p = p.astype(bool)
    points_1 = points[p]
    points_0 = points[~p]
    p = p.astype(float)
    sample_number_1 = int(p.mean() * sample_number)
    sample_number_0 = sample_number - sample_number_1

    kmeans = KMeans(n_clusters=sample_number_1, random_state=0, n_init='auto')
    kmeans.fit(points_1)
    intensity = np.bincount(kmeans.labels_, minlength=sample_number_1)
    x = kmeans.cluster_centers_[:, 0] * (W - 1)
    y = kmeans.cluster_centers_[:, 1] * (H - 1)
    t = kmeans.cluster_centers_[:, 2] / scale_t


    kmeans = KMeans(n_clusters=sample_number_0, random_state=0, n_init='auto')
    kmeans.fit(points_0)

    intensity = np.concatenate((intensity, np.bincount(kmeans.labels_, minlength=sample_number_0)))
    x = np.concatenate((x, kmeans.cluster_centers_[:, 0] * (W - 1)))
    y = np.concatenate((y, kmeans.cluster_centers_[:, 1] * (H - 1)))
    t = np.concatenate((t, kmeans.cluster_centers_[:, 2] / scale_t))
    p = np.zeros(sample_number, dtype=bool)
    p[0: sample_number_1] = True

    indices = np.argsort(t)

    x, y, t, p, intensity = x[indices], y[indices], t[indices], p[indices], intensity[indices]

    return x, y, t, p, intensity


class EventNPDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: str,
                 repeats: int = 1):
        '''
        :param training: is the train set
        :type training: bool
        :param root: the directory where the dataset is stored
        :type root: str
        :param sample_number: number of events to sample. Note that the number of events in one sample can be smaller than sample_number. In this case, if pad == False, then the number of returned events is less than sample_number, and the dataloader should pad events
        :type sample_number: int
        :param sampler: which sampler to use
        :type sampler: str





        root/
        ├── class_x
        │   ├── xxx.npz
        │   ├── xxy.npz
        │   └── ...
        │       └── xxz.npz
        └── class_y
            ├── 123.npz
            ├── nsdf3.npz
            └── ...
            └── asd932_.npz
        '''
        super().__init__(root=root,
                         loader=None,
                         extensions=('npz', 'npy'),
                         transform=None,
                         target_transform=None,
                         is_valid_file=None,
                         allow_empty=False)

        self.training = training

        self.rng = np.random.default_rng(0)  # will be reset by the dataloader
        self.sample_number = sample_number

        self.sampler_str = sampler
        if sampler is None or sampler.lower() == 'none':
            self.sampler = None

        elif sampler == 'random_sample':
            self.sampler = random_sample
        else:
            raise NotImplementedError(random_sample)


        self.repeats = repeats
        if repeats > 1:
            self.samples = self.samples * repeats


    def read_npz_by_key(self, opened_npz, k: str):
        return opened_npz[k]

    def __len__(self) -> int:

        return len(self.samples)

    @staticmethod
    def event_size():
        P = 0
        H = 0
        W = 0
        return P, H, W

    @staticmethod
    def num_classes():
        return -1

    def __getitem__(self, i: int):

        path, label = self.samples[i]

        sample = np.load(path, mmap_mode='c')
        t = self.read_npz_by_key(sample, 't')
        y = self.read_npz_by_key(sample, 'y')
        x = self.read_npz_by_key(sample, 'x')
        p = self.read_npz_by_key(sample, 'p')
        n = t.shape[0]
        intensity = None
  
        if self.sampler is not None:
            x, y, t, p = self.sampler(self.rng, x, y, t, p, self.sample_number)

        else:
            try:
                intensity = self.read_npz_by_key(sample, 'intensity')
            except ValueError:
                pass


        t = np.ascontiguousarray(t)
        t -= t.flat[0]
        t = torch.from_numpy(t).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        p = torch.from_numpy(np.ascontiguousarray(p)).float()

        if intensity is not None:
            intensity = torch.from_numpy(np.ascontiguousarray(intensity)).float()

        if self.repeats > 1 and not self.training:
            # 测试时用来计算std
            return x, y, t, p, intensity, label, i
        else:
            return x, y, t, p, intensity, label



def event_collate_fun_with_padding(batch: list):
    x = []
    y = []
    t = []
    p = []
    intensity = []
    label = []
    indices = []
    min_len = None
    max_len = None
    for i in range(len(batch)):
        b = batch[i]
        x.append(b[0])
        y.append(b[1])
        t.append(b[2])
        p.append(b[3])
        if b[4] is not None:
            intensity.append(b[4])
        label.append(b[5])
        if len(b) == 7:
            indices.append(b[6]) # indices
        l = b[0].numel()
        if i == 0:
            min_len = l
            max_len = l
        else:
            min_len = min(min_len, l)
            max_len = max(max_len, l)

    if min_len == max_len:
        x = torch.stack(x)
        y = torch.stack(y)
        t = torch.stack(t)
        p = torch.stack(p)
        if len(intensity) > 0:
            intensity = torch.stack(intensity)
        else:
            intensity = None
        valid_mask = torch.ones(x.shape, dtype=torch.bool)

    else:
        x = pad_sequence(x, batch_first=True, padding_value=-1)
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        t = pad_sequence(t, batch_first=True, padding_value=-1)
        p = pad_sequence(p, batch_first=True, padding_value=-1)
        if len(intensity) > 0:
            intensity = pad_sequence(intensity, batch_first=True, padding_value=-1)
        else:
            intensity = None

        valid_mask = (p != -1)

    label = torch.as_tensor(label, dtype=torch.long)
    if len(indices) == 0:
        return (x, y, t, p), intensity, valid_mask, label
    else:
        return (x, y, t, p), intensity, valid_mask, label, torch.as_tensor(indices, dtype=torch.long)




if __name__ == '__main__':
    pass