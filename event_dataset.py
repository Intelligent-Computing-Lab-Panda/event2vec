import math
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from torch.nn.utils.rnn import pad_sequence
import lightning
import h5py



def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.initial_seed()
    worker_rng = np.random.default_rng(worker_seed)
    worker_info.dataset.rng = worker_rng


def random_sample(rng: np.random.Generator, n: int, sample_number: int):
    indices = rng.choice(np.arange(n), sample_number, replace=False)
    indices.sort()
    return indices


def random_chunk_sample(rng: np.random.Generator, n: int, sample_number: int, chunk_size: int):
    n_chunk = sample_number // chunk_size
    indices = rng.choice(np.arange(n - chunk_size), n_chunk, replace=False)
    indices.sort()
    indices = np.expand_dims(indices, 1) + np.arange(chunk_size)
    indices = indices.reshape(-1)
    return indices


def uniform_sample(rng: np.random.Generator, n: int, sample_number: int):
    indices = rng.integers(low=0, high=n - 1, size=sample_number)
    indices.sort()
    return indices


def deterministic_uniform_sample(rng: np.random.Generator, n: int, sample_number: int):
    return np.linspace(0, n - 1, sample_number, dtype=np.long)


def gaussian_sample(rng: np.random.Generator, n: int, sample_number: int):
    indices = rng.normal(loc=n / 2, scale=n / 4, size=(sample_number,))
    np.clip(indices, 0, n - 1, out=indices).sort()
    indices = indices.astype(np.long)
    return indices


def multivariate_gaussian_sample(rng: np.random.Generator, n: int, sample_number: int):
    '''
    We split [0, n] to sample_number intervals:
        [0, n / sample_number]
        [n / sample_number, 2 * n / sample_number]
        ...
        [(sample_number - 1) * n / sample_number, sample_number * n / sample_number]

    In each interval i, the centre is (i + 0.5) * n / sample_number
    We generate a gaussian distribution with mean = (i + 0.5) * n / sample_number and std = n / sample_number / 4

    '''
    mean = np.arange(sample_number) * n / sample_number
    std = np.full(sample_number, n / sample_number / 4)
    indices = rng.normal(loc=mean, scale=std, size=(sample_number,))
    np.clip(indices, 0, n - 1, out=indices).sort()
    indices = indices.astype(np.long)
    return indices


samplers = {
    'random_sample': random_sample,
    'random_chunk_sample': random_chunk_sample,
    'uniform_sample': uniform_sample,
    'gaussian_sample': gaussian_sample,
    'deterministic_uniform_sample': deterministic_uniform_sample,
    'multivariate_gaussian_sample': multivariate_gaussian_sample,
}

def create_sampler_fun(sampler: str):
    if sampler.startswith('random_chunk_sample'):
        # random_chunk_sample_10 means chunk_size == 10
        chunk_size = int(sampler.split('_')[-1])
        return lambda rng, n, sample_number: random_chunk_sample(rng, n, sample_number, chunk_size)
    else:
        return samplers[sampler]

def test_samplers():
    rng = np.random.default_rng(0)
    n = 10000
    sample_number = 10
    print('random_sample\n', random_sample(rng, n, sample_number))
    print('uniform_sample\n', uniform_sample(rng, n, sample_number))
    print('deterministic_uniform_sample\n', deterministic_uniform_sample(rng, n, sample_number))
    print('gaussian_sample\n', gaussian_sample(rng, n, sample_number))
    print('multivariate_gaussian_sample\n', multivariate_gaussian_sample(rng, n, sample_number))
    print('random_chunk_sample\n', random_chunk_sample(rng, n, sample_number, chunk_size=2))


class EventNPDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, norm_t: str,
                 repeats: int = 1):
        '''
        :param training: is the train set
        :type training: bool
        :param root: the directory where the dataset is stored
        :type root: str
        :param sample_number: number of events to sample
        :type sample_number: int
        :param sampler: which sampler to use
        :type sampler: types.FunctionType
        :param norm_t: type of normalization, which can be
                        'none': return the original t (long)
                        'sample-wise': return t / t[-1] (float)
                        'dataset-wise': return t / t_max (float), where t_max is max(t[-1]) over the dataset.
                        Note that the test set should t_max from the train set
        :type norm_t: str
        :param repeats: how many times to repeat the dataset. Note that the sample is randomly, and it is better to repeat samples for the test set
        :type repeats: int


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
                         extensions=('npy', 'npz'),
                         transform=None,
                         target_transform=None,
                         is_valid_file=None,
                         allow_empty=False)

        self.training = training

        self.rng = np.random.default_rng(0)  # will be reset by the dataloader
        self.sample_number = sample_number
        self.sampler = sampler

        assert norm_t in ('none', 'sample-wise', 'dataset-wise'), norm_t
        self.norm_t = norm_t
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    @staticmethod
    def event_size():
        P = 0
        H = 0
        W = 0
        return P, H, W

    def statistics(self):
        t_max = 0
        t_min = math.inf
        n_max = 0
        n_min = math.inf
        n_events = 0
        for i in tqdm.trange(len(self.samples)):
            path, label = self.samples[i]
            sample = np.load(path, mmap_mode='r')
            t = sample['t']
            n = t.shape[0]
            t = t[-1] - t[0]
            t_max = max(t_max, t)
            t_min = min(t_min, t)
            n_events += n
            n_max = max(n_max, n)
            n_min = min(n_min, n)
        return t_min, t_max, n_min, n_max, n_events

    def __getitem__(self, i: int):
        i = i % len(self.samples)
        path, label = self.samples[i]

        sample = np.load(path, mmap_mode='r')
        t = sample['t']
        y = sample['y']
        x = sample['x']
        p = sample['p']
        n = t.shape[0]
        if self.sample_number <= n:
            indices = self.sampler(self.rng, n, self.sample_number)
            t = t[indices]
            y = y[indices]
            x = x[indices]
            p = p[indices]

        t -= t[0]
        if self.norm_t == 'sample-wise':
            t = t.astype(np.float32)
            t /= t[-1]
            t = torch.from_numpy(t).float()
        elif self.norm_t == 'dataset-wise':
            t = t.astype(np.float32)
            t /= self.t_max
            t = torch.from_numpy(t).float()
        elif self.norm_t == 'none':
            t = torch.from_numpy(t).long()
        else:
            raise NotImplementedError(self.norm_t)

        y = torch.from_numpy(y).long()
        x = torch.from_numpy(x).long()
        p = torch.from_numpy(p).long()

        return x, y, t, p, label


def event_collate_fun(batch: list):
    x = []
    y = []
    t = []
    p = []
    label = []

    for i in range(len(batch)):
        b = batch[i]
        x.append(b[0])
        y.append(b[1])
        t.append(b[2])
        p.append(b[3])
        label.append(b[4])

    x = torch.stack(x)
    y = torch.stack(y)
    t = torch.stack(t)
    p = torch.stack(p)

    label = torch.as_tensor(label, dtype=torch.long)

    return (x, y, t, p), torch.ones_like(p, dtype=torch.bool), label


def event_collate_fun_with_padding(batch: list):
    x = []
    y = []
    t = []
    p = []
    label = []

    for i in range(len(batch)):
        b = batch[i]
        x.append(b[0])
        y.append(b[1])
        t.append(b[2])
        p.append(b[3])
        label.append(b[4])

    x = pad_sequence(x, batch_first=True, padding_value=-1)
    y = pad_sequence(y, batch_first=True, padding_value=-1)
    t = pad_sequence(t, batch_first=True, padding_value=-1)
    p = pad_sequence(p, batch_first=True, padding_value=-1)

    valid_mask = (p != -1)

    label = torch.as_tensor(label, dtype=torch.long)

    return (x, y, t, p), valid_mask, label



class ASLDVS(EventNPDataset):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: types.FunctionType, norm_t: str,
                 repeats: int = 1):
        super().__init__(training=training, root=root, sample_number=sample_number, sampler=sampler, norm_t=norm_t,
                         repeats=repeats)

        train_ratio = 0.8
        num_classes = 24

        samples_in_class = []
        for i in range(num_classes):
            samples_in_class.append([])

        for i in range(len(self.samples)):
            path, label = self.samples[i]
            samples_in_class[label].append([path, label])

        self.samples.clear()
        for i in range(num_classes):
            pos = int(len(samples_in_class[i]) * train_ratio)
            if self.training:
                self.samples.extend(samples_in_class[i][0:pos])
            else:
                self.samples.extend(samples_in_class[i][pos:])

        del self.classes
        del self.class_to_idx
        del self.targets

        self.t_max = 521217

    @staticmethod
    def event_size():
        P = 2
        H = 180
        W = 240
        return P, H, W

    def statistics(self):
        # t_min, t_max, n_min, n_max, n_events
        if self.training:
            return 99, 521217, 195, 470435, 2268004196
        else:
            return 2684, 521210, 198, 466524, 569412358





class EventDataModule(lightning.LightningDataModule):
    def __init__(self, name: str, root: str, sample_number: int, sampler: str, norm_t: str, batch_size: int,
                 num_workers: int):
        super().__init__()
        self.name = name
        self.root = root
        self.sample_number = sample_number

        self.sampler = create_sampler_fun(sampler)
        self.norm_t = norm_t
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if self.name == 'asl_dvs':
            dts_class = ASLDVS
        else:
            raise NotImplementedError(self.name)

        self.train_set = dts_class(training=True, root=self.root, sample_number=self.sample_number,
                                    sampler=self.sampler, norm_t=self.norm_t)
        self.val_set = dts_class(training=False, root=self.root, sample_number=self.sample_number,
                                  sampler=self.sampler, norm_t=self.norm_t, repeats=4)

    def train_dataloader(self):
        if self.train_set.statistics()[2] >= self.sample_number:
            collate_fn = event_collate_fun
        else:
            collate_fn = event_collate_fun_with_padding
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, drop_last=True,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=seed_worker,
                                           persistent_workers=True,
                                           prefetch_factor=4)

    def val_dataloader(self):
        if self.val_set.statistics()[2] >= self.sample_number:
            collate_fn = event_collate_fun
        else:
            collate_fn = event_collate_fun_with_padding

        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, drop_last=False,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           collate_fn=collate_fn,
                                           worker_init_fn=seed_worker,
                                           persistent_workers=True,
                                           prefetch_factor=4)



if __name__ == '__main__':

    pass




