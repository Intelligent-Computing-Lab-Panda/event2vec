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
from torch.nn.utils.rnn import pad_sequence
import lightning
import h5py
import numba
import importlib
import heapq
from concurrent.futures import ThreadPoolExecutor




# download gen1 from https://kdrive.infomaniak.com/app/share/975517/0ccd6970-a11a-4a85-8f6b-db34f3556f21/files/37
# save to /vast/palmer/pi/panda/wf282
def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.initial_seed()
    worker_rng = np.random.default_rng(worker_seed)
    worker_info.dataset.rng = worker_rng


def pad_events(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, n: int, rng: np.random.Generator):
    # pad the number of events to n by random sampling
    n_padding = n - x.size
    assert n_padding > 0

    indices = rng.choice(x.size, n_padding, replace=True)

    events = np.stack((x, y, t, p), axis=1)  # 形状为 (x.size, 4)

    padded_events = np.concatenate((events, events[indices]), axis=0)

    sorted_indices = np.argsort(padded_events[:, 2])
    padded_events = padded_events[sorted_indices]

    return padded_events[:, 0], padded_events[:, 1], padded_events[:, 2], padded_events[:, 3]

@numba.jit(nopython=True)
def event_temporal_smooth(x_: np.ndarray, y_: np.ndarray, t_: np.ndarray, p_: np.ndarray, P: int, H: int, W: int, half_life: float):
    assert 0 < half_life <= 1
    leak = - math.log(half_life) / half_life
    v = np.zeros((P, H, W), dtype=np.float32)
    t_last = np.zeros((P, H, W), dtype=np.float32)
    v_out = np.zeros(p_.shape, dtype=np.float32)
    scale = t_[-1] - t_[0]
    for i in range(x_.size):
        p = p_[i]
        y = y_[i]
        x = x_[i]
        t = t_[i]
        delta_t = (t_last[p][y][x] - t) / scale
        # -1 <= delta_t <= 0
        v[p][y][x] *= math.exp(delta_t * leak)
        v[p][y][x] += 1.
        v_out[i] = v[p][y][x] * (p * 2 - 1)
        t_last[p][y][x] = t
    return v_out


@numba.jit(nopython=True)
def leaky_integrate_and_fire_event_temporal_filter(x_: np.ndarray, y_: np.ndarray, t_: np.ndarray, p_: np.ndarray, P:int, H: int, W: int, half_life: float, n: int):
    assert n < x_.shape[0]
    assert 0 < half_life <= 1
    leak = - math.log(half_life) / half_life

    event_out = np.zeros((n, 4), dtype=np.long)
    v = np.zeros((P, H, W), dtype=np.float32)
    t_last = np.zeros((P, H, W), dtype=np.float32)

    index = 0
    left = 0. # x_.shape[0] events
    right = float(x_.shape[0]) # 0 events
    while True:
        i = 0
        v_th = (left + right) / 2.

        while True:
            p = p_[i]
            y = y_[i]
            x = x_[i]
            t = t_[i]
            delta_t = (t_last[p][y][x] - t) / t_[-1]
            # -1 <= delta_t <= 0
            v[p][y][x] *= math.exp(delta_t * leak)
            v[p][y][x] += 1.
            if v[p][y][x] >= v_th:
                event_out[index] = (x, y, t, p)
                t_last[p][y][x] = t
                index += 1
                if index == n:
                    break
                v[p][y][x] -= v_th

            i += 1
            if i == x_.shape[0]:
                break

        if index == n:
            if i == x_.shape[0] or right - left < 1e-5:
                # just ok!
                return event_out
            else:
                assert i < x_.shape[0]
                # v_th is too small!
                left = v_th
        else:
            # v_th is too large!
            right = v_th

        index = 0
        v.fill(0.)
        t_last.fill(0.)



@numba.jit(nopython=True)
def leaky_integrate_and_fire_event_joint_filter(x_: np.ndarray, y_: np.ndarray, t_: np.ndarray, p_: np.ndarray, P:int, H: int, W: int, half_life_x: float, half_life_y: float, half_life_t: float, n: int):
    assert n < x_.shape[0]
    assert 0 < half_life_x <= 1
    assert 0 < half_life_y <= 1
    assert 0 < half_life_t <= 1
    leak_x = - math.log(half_life_x) / half_life_x
    leak_y = - math.log(half_life_y) / half_life_y
    leak_t = - math.log(half_life_t) / half_life_t

    event_out = np.zeros((n, 4), dtype=np.long)
    indices = np.zeros(n, dtype=np.long)
    v = np.zeros((P, H, W), dtype=np.float32)
    t_last = None
    y_last = None
    x_last = None

    index = 0
    left = 0. # x_.shape[0] events
    right = float(x_.shape[0]) # 0 events
    while True:
        i = 0
        v_th = (left + right) / 2.

        while True:
            p = p_[i]
            y = y_[i]
            x = x_[i]
            t = t_[i]
            if t_last is None:
                delta_t = 0.
            else:
                delta_t = - abs(t_last - t) / t_[-1]
            if y_last is None:
                delta_y = 0.
            else:
                delta_y = - abs(y_last - y) / (H - 1)
            if x_last is None:
                delta_x = 0.
            else:
                delta_x = - abs(x_last - x) / (W - 1)


            v[p][y][x] *= math.exp(delta_t * leak_t + delta_y * leak_y + delta_x * leak_x)

            v[p][y][x] += 1.
            if v[p][y][x] >= v_th:
                event_out[index] = (x, y, t, p)
                indices[index] = i
                x_last = x
                y_last = y
                t_last = t
                index += 1
                if index == n:
                    break
                v[p][y][x] -= v_th

            i += 1
            if i == x_.shape[0]:
                break

        if index == n:
            if i == x_.shape[0] or right - left < 1e-5:
                # just ok!
                return event_out, indices
            else:
                assert i < x_.shape[0]
                # v_th is too small!
                left = v_th
        else:
            # v_th is too large!
            right = v_th

        index = 0
        v.fill(0.)
        t_last = None
        y_last = None
        x_last = None





def lif_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int, P:int, H: int, W: int, half_life_x: float, half_life_y: float, half_life_t: float):
    n = x.size
    assert n > sample_number
    events, indices = leaky_integrate_and_fire_event_joint_filter(x_=x.astype(np.long), y_=y.astype(np.long), t_=t.astype(np.long), p_=p.astype(np.long), P=P, H=H, W=W, half_life_x=half_life_x, half_life_y=half_life_y, half_life_t=half_life_t, n=sample_number)
    return events[:, 0], events[:, 1], events[:, 2], events[:, 3], indices


def random_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int):
    n = x.size
    indices = rng.choice(n, sample_number, replace=False)
    indices.sort()
    return x[indices], y[indices], t[indices], p[indices], indices

def random_lif_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number_pre: int, sample_number_post: int, P:int, H: int, W: int, half_life_x: float, half_life_y: float, half_life_t: float):
    n = x.size
    if n <= sample_number_pre:
        if n <= sample_number_post:
            print(n, sample_number_post)
            exit(-1)

        return random_sample(rng=rng, x=x, y=y, t=t, p=p, sample_number=sample_number_post)

    else:
        x, y, t, p, indices_pre = random_sample(rng=rng, x=x, y=y, t=t, p=p, sample_number=sample_number_pre)

        x, y, t, p, indices_post = lif_sample(rng=rng, x=x, y=y, t=t, p=p, sample_number=sample_number_post, P=P, H=H, W=W, half_life_x=half_life_x, half_life_y=half_life_y, half_life_t=half_life_t)

        return x, y, t, p, indices_pre[indices_post]

def temporal_random_sample(
        rng: np.random.Generator,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        sample_number: int
):
    dt = np.diff(t)
    dt = dt.astype(np.float64)
    dt[dt == 0] = 1e-9
    dt = dt / dt.sum()
    x = x[:-1]
    y = y[:-1]
    t = t[:-1]
    p = p[:-1]
    n = dt.size

    indices = rng.choice(n, sample_number, replace=False, p=dt)
    indices.sort()

    return x[indices], y[indices], t[indices], p[indices]


def block_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, block_number: int, block_size: int, interval:int):
    n = x.size
    '''
    采样block_number个block

    每个block内是连续的block_size个event，其间隔为interval
        它们在每个block内的索引为
        0, interval, interval * 2, ..., interval * (block_size - 1)
        
    最大的start_indices应该满足
    start_indices + interval * (block_size - 1) = n - 1
    因此start_indices最大为 n - 1 - interval * (block_size - 1)
    
    '''
    assert block_number * block_size <= n
    while True:
        max_start_index = n - interval * (block_size - 1)
        if interval == 1:
            break
        elif max_start_index < block_number:
            interval -= 1


    start_indices = rng.choice(max_start_index, block_number, replace=False)

    start_indices.sort()
    offsets = np.arange(block_size) * interval

    indices = start_indices.reshape((-1, 1)) + offsets
    indices = indices.flatten()
    return x[indices], y[indices], t[indices], p[indices]









if importlib.util.find_spec('fpsample'):
    import fpsample
    def farthest_point_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int, H: int, W: int):

        t_max = t[-1]

        x = x.astype(np.float32)
        x /= (W - 1)
        y = y.astype(np.float32)
        y /= (H - 1)

        t -= t[0]
        t = t.astype(np.float32)
        t /= t_max

        x *= 2.
        x -= 1.

        y *= 2.
        y -= 1.

        t *= 2.
        t -= 1.

        points = np.stack((x, y, t), axis=1)
        indices = fpsample.bucket_fps_kdline_sampling(points, sample_number, h=7)
        x = x[indices]
        y = y[indices]
        t = t[indices]

        x += 1.
        x /= 2.

        y += 1.
        y /= 2.

        t += 1.
        t /= 2.

        x *= (W - 1)
        x = x.astype(np.long)

        y *= (H - 1)
        y = y.astype(np.long)

        t *= t_max
        t = t.astype(np.long)



        p = p[indices]

        return x, y, t, p


def cube_mean(y, x, t, cube_H, cube_W, cube_T, index):
    xm = torch.zeros(cube_T * cube_H * cube_W)

    xm.scatter_reduce_(dim=0, index=index, src=x.float(), reduce='mean')

    ym = torch.zeros(cube_T * cube_H * cube_W)
    ym.scatter_reduce_(dim=0, index=index, src=y.float(), reduce='mean')

    tm = torch.zeros(cube_T * cube_H * cube_W)
    tm.scatter_reduce_(dim=0, index=index, src=t.float(), reduce='mean')

    counts = torch.zeros(cube_T * cube_H * cube_W, dtype=torch.long, device=index.device)
    counts.scatter_add_(dim=0, index=index, src=torch.ones_like(index, dtype=torch.long))
    valid_mask = counts > 0

    return ym[valid_mask], xm[valid_mask], tm[valid_mask]

def cube_mean_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int, H:int, W:int, h: int, w: int, tau: int):
    t = t - np.min(t)
    T = np.max(t) + 1

    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()
    t = torch.from_numpy(t).long()
    p = torch.from_numpy(p).bool()

    ix = x // w
    iy = y // h
    it = t // tau

    cube_H = int(math.ceil(H / h))
    cube_W = int(math.ceil(W / w))
    cube_T = int(math.ceil(T / tau))

    index = it * cube_H * cube_W + iy * cube_W + ix

    y1, x1, t1 = cube_mean(y[p], x[p], t[p], cube_H, cube_W, cube_T, index[p])
    p0 = ~p
    y0, x0, t0 = cube_mean(y[p0], x[p0], t[p0], cube_H, cube_W, cube_T, index[p0])

    x = torch.cat((x0, x1)).round_().clamp_(0, W - 1).long()
    y = torch.cat((y0, y1)).round_().clamp_(0, H - 1).long()
    t = torch.cat((t0, t1)).round_().clamp_(0, T - 1).long()
    p = torch.cat((torch.zeros_like(x0, dtype=torch.bool), torch.ones_like(x1, dtype=torch.bool)))

    i = torch.argsort(t)
    x = x[i].numpy()
    y = y[i].numpy()
    t = t[i].numpy()
    p = p[i].numpy()
    if x.size > sample_number:
        return random_sample(rng, x, y, t, p, sample_number)
    else:
        return x, y, t, p


def uniform_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int):
    indices = rng.integers(low=0, high=x.size - 1, size=sample_number)
    indices.sort()
    return x[indices], y[indices], t[indices], p[indices]


def deterministic_uniform_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int):
    indices = np.linspace(0, x.size - 1, sample_number, dtype=np.long)
    return x[indices], y[indices], t[indices], p[indices]


def gaussian_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int):
    n = x.size
    indices = rng.normal(loc=n / 2, scale=n / 4, size=(sample_number,))
    np.clip(indices, 0, n - 1, out=indices).sort()
    indices = indices.astype(np.long)
    return x[indices], y[indices], t[indices], p[indices]


def multivariate_gaussian_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, sample_number: int):
    '''
    We split [0, n] to sample_number intervals:
        [0, n / sample_number]
        [n / sample_number, 2 * n / sample_number]
        ...
        [(sample_number - 1) * n / sample_number, sample_number * n / sample_number]

    In each interval i, the centre is (i + 0.5) * n / sample_number
    We generate a gaussian distribution with mean = (i + 0.5) * n / sample_number and std = n / sample_number / 4

    '''
    n = x.size
    mean = np.arange(sample_number) * n / sample_number
    std = np.full(sample_number, n / sample_number / 4)
    indices = rng.normal(loc=mean, scale=std, size=(sample_number,))
    np.clip(indices, 0, n - 1, out=indices).sort()
    indices = indices.astype(np.long)
    return x[indices], y[indices], t[indices], p[indices]


samplers = {
    'random_sample': random_sample,
    'uniform_sample': uniform_sample,
    'gaussian_sample': gaussian_sample,
    'deterministic_uniform_sample': deterministic_uniform_sample,
    'multivariate_gaussian_sample': multivariate_gaussian_sample,
    'temporal_random_sample': temporal_random_sample
}

def create_sampler_fun(s: str):
    if s in samplers:
        return samplers[s]
    elif s.startswith('farthest_point_sample-'):
        # farthest_point_sample-{H,W}
        s = s.split('-')
        HW = s[1].split(',')
        H = int(HW[0])
        W = int(HW[1])
        return lambda rng, x, y, t, p, sample_number: farthest_point_sample(rng, x, y, t, p, sample_number, H, W)
    elif s.startswith('cube_mean_sample-'):
        # cube_mean_sample-{H,W}-{h,w,tau}
        s = s.split('-')
        temp = s[1].split(',')
        H = int(temp[0])
        W = int(temp[1])

        temp = s[2].split(',')
        h = int(temp[0])
        w = int(temp[1])
        tau = int(temp[2])
        return lambda rng, x, y, t, p, sample_number: cube_mean_sample(rng, x, y, t, p, sample_number, H, W, h, w, tau)
    elif s.startswith('lif-'):
        # lif-{P,H,W}-{hx,hy,ht}
        s = s.split('-')
        temp = s[1].split(',')
        P = int(temp[0])
        H = int(temp[1])
        W = int(temp[2])

        temp = s[2].split(',')
        hx = float(temp[0])
        hy = float(temp[1])
        ht = float(temp[2])
        return lambda rng, x, y, t, p, sample_number: lif_sample(rng=rng, x=x, y=y, t=t, p=p, sample_number=sample_number, P=P, H=H, W=W, half_life_x=hx, half_life_y=hy, half_life_t=ht)

    elif s.startswith('block_sample-'):
        # block_sample-block_size-interval
        s = s.split('-')
        block_size = int(s[1])
        interval = int(s[2])
        return lambda rng, x, y, t, p, sample_number: block_sample(rng, x, y, t, p, sample_number // block_size, block_size, interval)

    elif s.startswith('random_lif_sample-'):
        # random_lif_sample-{P, H, W}-{hx, hy, ht}-{sample_number_pre}
        s = s.split('-')
        temp = s[1].split(',')
        P = int(temp[0])
        H = int(temp[1])
        W = int(temp[2])

        temp = s[2].split(',')
        hx = float(temp[0])
        hy = float(temp[1])
        ht = float(temp[2])

        sample_number_pre = int(s[3])
        return lambda rng, x, y, t, p, sample_number: random_lif_sample(rng=rng, x=x, y=y, t=t, p=p,
                                                                 sample_number_pre=sample_number_pre,
                                                              sample_number_post=sample_number,
                                                                        P=P, H=H, W=W,
                                                                 half_life_x=hx, half_life_y=hy, half_life_t=ht)


    else:
        raise NotImplementedError

def test_samplers():
    rng = np.random.default_rng(0)
    n = 10000
    sample_number = 10
    print('random_sample\n', random_sample(rng, n, sample_number))
    print('uniform_sample\n', uniform_sample(rng, n, sample_number))
    print('deterministic_uniform_sample\n', deterministic_uniform_sample(rng, n, sample_number))
    print('gaussian_sample\n', gaussian_sample(rng, n, sample_number))
    print('multivariate_gaussian_sample\n', multivariate_gaussian_sample(rng, n, sample_number))




class EventNPDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: str, norm_t: str, full_sample:bool, return_index:bool, repeats:int=1):
        '''
        :param training: is the train set
        :type training: bool
        :param root: the directory where the dataset is stored
        :type root: str
        :param sample_number: number of events to sample. Note that the number of events in one sample can be smaller than sample_number. In this case, if pad == False, then the number of returned events is less than sample_number, and the dataloader should pad events
        :type sample_number: int
        :param sampler: which sampler to use
        :type sampler: str
        :param norm_t: type of normalization, which can be
                        'none': return the original t (long)
                        'sample-wise': return t / t[-1] (float)
                        'dataset-wise': return t / t_max (float), where t_max is max(t[-1]) over the dataset.
                        Note that the test set should t_max from the train set
        :type norm_t: str




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
        self.sampler = create_sampler_fun(sampler)



        assert norm_t in ('none', 'sample-wise', 'dataset-wise'), norm_t
        self.norm_t = norm_t
        self.repeats = repeats
        if repeats > 1:
            self.samples = self.samples * repeats


        self.full_sample = full_sample
        self.return_index = return_index
        if return_index:
            assert full_sample
        if full_sample:
            samples_split = []
            sampled_indices = {}
            if return_index:
                split_index_to_index = {}
                n_split = []
            for i in range(len(self.samples)):
                path, label = self.samples[i]
                sampled_indices[path] = set()

                sample = np.load(path, mmap_mode='c')
                t = self.read_npz_by_key(sample, 't')
                n = t.shape[0]
                if return_index:
                    n_split.append(math.ceil(n / sample_number))
                for _ in range(math.ceil(n / sample_number)):
                    if return_index:
                        split_index_to_index[len(samples_split)] = i
                    samples_split.append((path, label))
            self.samples_split = samples_split
            self.sampled_indices = sampled_indices
            if return_index:
                self.split_index_to_index = split_index_to_index
                self.n_split = n_split


    def all_labels(self):
        labels = []
        for i in range(len(self.samples)):
            path, label = self.samples[i]
            labels.append(label)
        return torch.as_tensor(labels, dtype=torch.long)

    def save_sample_to_dir(self, i, out_dir):
        x, y, t, p, label = self.__getitem__(i)
        label_dir = os.path.join(out_dir, str(label))
        fname = os.path.basename(self.samples[i][0])
        fname = os.path.join(label_dir, fname)
        np.savez(
            fname,
            x=x.numpy(), y=y.numpy(), t=t.numpy(), p=p.numpy()
        )
        print('save', fname)

    def save_to_dir(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for label in range(self.num_classes()):
            label_dir = os.path.join(out_dir, str(label))
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
        with ThreadPoolExecutor(max_workers=36) as tpe:
            sub_threads = []

            for i in range(len(self.samples)):
                sub_threads.append(
                    tpe.submit(self.save_sample_to_dir, i, out_dir))
            for sub_thread in sub_threads:
                if sub_thread.exception():
                    print(sub_thread.exception())
                    exit(-1)




    def read_npz_by_key(self, opened_npz, k:str):
        return opened_npz[k]


    def __len__(self) -> int:
        if self.full_sample:
            return len(self.samples_split)
        else:
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

    def statistics(self):
        t_max = 0
        t_min = math.inf
        n_max = 0
        n_min = math.inf
        n_events = 0
        for i in tqdm.trange(len(self.samples)):
            path, label = self.samples[i]
            sample = np.load(path, mmap_mode='r')
            t = self.read_npz_by_key(sample, 't')
            n = t.shape[0]
            t = t[-1] - t[0]
            t_max = max(t_max, t)
            t_min = min(t_min, t)
            n_events += n
            n_max = max(n_max, n)
            n_min = min(n_min, n)
        return {'t_min': t_min, 't_max': t_max, 'n_min': n_min, 'n_max': n_max, 'n_events': n_events}


    def __getitem__(self, i: int):
        if self.full_sample:
            path, label = self.samples_split[i]
        else:
            path, label = self.samples[i]

        sample = np.load(path, mmap_mode='c')
        t = self.read_npz_by_key(sample, 't')
        t = np.astype(t, np.int64)
        y = self.read_npz_by_key(sample, 'y')
        x = self.read_npz_by_key(sample, 'x')
        p = self.read_npz_by_key(sample, 'p')
        n = t.shape[0]




        if self.sample_number < n:
            if self.full_sample:
                if len(self.sampled_indices[path]) == 0:
                    x, y, t, p, indices = self.sampler(self.rng, x, y, t, p, self.sample_number)
                    self.sampled_indices[path].update(set(indices))

                else:
                    full_indices = set(np.arange(n).tolist())
                    rest_indices = np.asarray(list(full_indices.difference(self.sampled_indices[path])))

                    if self.sample_number < rest_indices.size:
                        rest_indices.sort()
                        x, y, t, p, indices = self.sampler(self.rng, x[rest_indices], y[rest_indices], t[rest_indices], p[rest_indices], self.sample_number)
                        self.sampled_indices[path].update(set(rest_indices[indices]))
                    else:
                        # 在之前已经使用过的数据中额外采样，以补足事件数量
                        padding_indices = self.rng.choice(np.asarray(list(self.sampled_indices[path])), self.sample_number - rest_indices.size, replace=False)
                        indices = np.concatenate((rest_indices, padding_indices))
                        indices.sort()
                        x = x[indices]
                        y = y[indices]
                        t = t[indices]
                        p = p[indices]
                        self.sampled_indices[path].update(set(rest_indices))


                if len(self.sampled_indices[path]) == n:
                    self.sampled_indices[path].clear()
            else:
                x, y, t, p, _ = self.sampler(self.rng, x, y, t, p, self.sample_number)

        # sample_number >= n 的情况不需要考虑，因为直接把所有事件都拿去用了

        t -= t.flat[0]
        if self.norm_t == 'sample-wise':
            t = t.astype(np.float32)
            t /= t[-1]
            t = torch.from_numpy(t).float()
        elif self.norm_t == 'dataset-wise':
            t = t.astype(np.float32)
            t /= self.t_max
            t = torch.from_numpy(t).float()
        elif self.norm_t == 'none':
            t = torch.from_numpy(t).float()
        else:
            raise NotImplementedError(self.norm_t)


        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        p = torch.from_numpy(np.ascontiguousarray(p)).float()
        if self.return_index:
            return x, y, t, p, label, self.split_index_to_index[i]
        else:
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


def event_collate_fun_with_padding_and_index(batch: list):
    x = []
    y = []
    t = []
    p = []
    label = []
    index = []

    for i in range(len(batch)):
        b = batch[i]
        x.append(b[0])
        y.append(b[1])
        t.append(b[2])
        p.append(b[3])
        label.append(b[4])
        index.append(b[5])


    x = pad_sequence(x, batch_first=True, padding_value=-1)
    y = pad_sequence(y, batch_first=True, padding_value=-1)
    t = pad_sequence(t, batch_first=True, padding_value=-1)
    p = pad_sequence(p, batch_first=True, padding_value=-1)

    valid_mask = (p != -1)

    label = torch.as_tensor(label, dtype=torch.long)

    index = torch.as_tensor(index, dtype=torch.long)
    return (x, y, t, p), valid_mask, (label, index)

def suffix_file_list(dir_path:str, suffix: str):
    # return the file name list of *.suffix in dir_path
    fnames = []
    for fname in os.listdir(dir_path):
        if os.path.splitext(fname)[1][1:] == suffix:
            fnames.append(os.path.join(dir_path, fname))
    return fnames



if __name__ == '__main__':
    pass