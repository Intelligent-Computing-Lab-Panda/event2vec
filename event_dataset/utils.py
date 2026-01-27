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
from sklearn.cluster import KMeans, MiniBatchKMeans
from numba import jit

from concurrent.futures import ThreadPoolExecutor



def seed_worker(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.initial_seed()
    worker_rng = np.random.default_rng(worker_seed)
    worker_info.dataset.rng = worker_rng
    if hasattr(worker_info.dataset, 'WORKER_FAISS_RES'):
        worker_info.dataset.WORKER_FAISS_RES = faiss.StandardGpuResources()
    

def random_sample(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                  sample_number: int):
    n = x.size
    if n > sample_number:
        indices = rng.choice(n, sample_number, replace=False)
        indices.sort()
        return x[indices], y[indices], t[indices], p[indices]
    else:
        return x, y, t, p

# 1024, 100, 0.7342713475227356 53.13
# 2048, 100, 0.7415075302124023, 74.29
# tol=0.1, 0.7511557936668396, 75.87
# max_iter=1, 0.7487437129020691, 69.63
def k_mean_cluster(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                  sample_number: int, H: int, W: int, scale_t: float, mini:bool=False, mini_batch_size:int=1024, mini_max_iter:int=100):
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

    if mini:
        kmeans = MiniBatchKMeans(n_clusters=sample_number_1, random_state=0, n_init='auto', batch_size=mini_batch_size, max_iter=mini_max_iter)
    else:
        kmeans = KMeans(n_clusters=sample_number_1, random_state=0, n_init='auto', max_iter=1)
    kmeans.fit(points_1)
    intensity = np.bincount(kmeans.labels_, minlength=sample_number_1)
    x = kmeans.cluster_centers_[:, 0] * (W - 1)
    y = kmeans.cluster_centers_[:, 1] * (H - 1)
    t = kmeans.cluster_centers_[:, 2] / scale_t




    if mini:
        kmeans = MiniBatchKMeans(n_clusters=sample_number_0, random_state=0, n_init='auto', batch_size=mini_batch_size, max_iter=mini_max_iter)
    else:
        kmeans = KMeans(n_clusters=sample_number_0, random_state=0, n_init='auto', max_iter=1)

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

import numpy as np

def add_noise_injection(real_events, k_ratio, sensor_width, sensor_height):
    """
    向事件流中注入模拟热噪声。

    :param real_events: shape [N_real, 4] 的 numpy 数组, 每一行是 (x, y, t, p)
    :param k_ratio: 噪声比例 (例如 0.5 表示 50%)
    :param sensor_width: 传感器宽度 (W)
    :param sensor_height: 传感器高度 (H)
    :return: 注入噪声并重新排序的事件流
    """

    N_real = real_events.shape[0]
    N_noise = int(k_ratio * N_real)

    if N_noise == 0:
        return real_events

    # 1. 获取原始数据的时间和空间范围
    t_min = real_events[0, 2]
    t_max = real_events[-1, 2] # 假设 real_events 已经按时间排序

    # 2. 生成 N_noise 个噪声事件
    # 随机 x 坐标
    noise_x = np.random.randint(0, sensor_width, size=N_noise)
    # 随机 y 坐标
    noise_y = np.random.randint(0, sensor_height, size=N_noise)
    # 随机 t 时间戳
    noise_t = np.random.uniform(t_min, t_max, size=N_noise)
    # 随机 p 极性 (在 {0, 1} 中选择)
    noise_p = np.random.choice([0, 1], size=N_noise)

    # 3. 组合成 [N_noise, 4] 的数组
    noise_events = np.stack([noise_x, noise_y, noise_t, noise_p], axis=1)

    # 4. 合并并重新排序
    noisy_events = np.concatenate([real_events, noise_events])
    
    # 关键：按时间戳 (第 3 列, 索引为 2) 重新排序
    noisy_events = noisy_events[noisy_events[:, 2].argsort()]

    return noisy_events

def add_spatial_jitter(events: np.ndarray, 
                       percentage: float, 
                       sigma: float, 
                       sensor_width: int, 
                       sensor_height: int) -> np.ndarray:
    """
    向事件流中添加“空间抖动”(Spatial Jitter)噪声。

    这种噪声会随机选择一部分事件，并将其 (x, y) 坐标
    按照 N(0, sigma^2) 的高斯分布进行随机偏移。

    参数:
    ----------
    events : np.ndarray
        输入的事件流 Numpy 数组, 形状为 [N, 4]。
        假设列的顺序是 (x, y, t, p)。
    percentage : float
        要修改的事件的百分比 (例如 0.2 表示 20%)。
    sigma : float
        用于生成 (dx, dy) 偏移量的高斯分布的标准差 (以像素为单位)。
    sensor_width : int
        传感器的宽度 (例如 346)，用于裁剪坐标。
    sensor_height : int
        传感器的高度 (例如 260)，用于裁剪坐标。

    返回:
    -------
    np.ndarray
        增加了空间抖动噪声的新事件流，形状与输入相同 [N, 4]。
    """
    
    # 0. 处理空事件流的边缘情况
    if len(events) == 0:
        return events

    # 1. 创建一个副本，以避免修改原始数据
    noisy_events = events.copy()
    
    # 2. 确定要修改的事件数量和索引
    num_events = len(events)
    num_to_modify = int(num_events * percentage)

    if num_to_modify == 0:
        # print("Warning: 0 events selected to modify.")
        return noisy_events

    # 随机选择 'num_to_modify' 个不重复的索引
    indices_to_modify = np.random.choice(num_events, size=num_to_modify, replace=False)

    # 3. 为被选中的事件生成高斯偏移量 (dx, dy)
    # dx 和 dy 都是从 N(0, sigma^2) 中采样
    dx = np.random.normal(loc=0.0, scale=sigma, size=num_to_modify)
    dy = np.random.normal(loc=0.0, scale=sigma, size=num_to_modify)

    # 4. 应用偏移、四舍五入并裁剪
    
    # 获取原始的 x, y 坐标 (使用浮点数进行中间计算以防溢出)
    original_x = noisy_events[indices_to_modify, 0].astype(np.float64)
    original_y = noisy_events[indices_to_modify, 1].astype(np.float64)

    # 添加偏移并四舍五入到最近的整数（像素）
    new_x = np.round(original_x + dx)
    new_y = np.round(original_y + dy)

    # 关键：裁剪坐标，使其保持在 [0, width-1] 和 [0, height-1] 的有效范围内
    new_x = np.clip(new_x, 0, sensor_width - 1)
    new_y = np.clip(new_y, 0, sensor_height - 1)

    # 5. 将修改后的坐标放回数组中
    # 将其转换回原始数据类型 (例如 uint16)
    noisy_events[indices_to_modify, 0] = new_x.astype(events.dtype)
    noisy_events[indices_to_modify, 1] = new_y.astype(events.dtype)

    return noisy_events


def add_polarity_noise(events: np.ndarray, percentage: float) -> np.ndarray:
    """
    向事件流中添加“极性噪声”(Polarity Noise)。

    这种噪声会随机选择一部分事件，并将其极性 (p) 进行反转。
    假设: 极性 p 使用 0 和 1 表示。
    反转逻辑: 0 -> 1, 1 -> 0 (即 p_new = 1 - p_old)

    参数:
    ----------
    events : np.ndarray
        输入的事件流 Numpy 数组, 形状为 [N, 4]。
        假设列的顺序是 (x, y, t, p)，且 p 在第4列 (索引3)。
    percentage : float
        要修改(反转极性)的事件的百分比 (例如 0.05 表示 5%)。

    返回:
    -------
    np.ndarray
        增加了极性噪声的新事件流，形状与输入相同 [N, 4]。
    """
    
    # 0. 处理空事件流的边缘情况
    if len(events) == 0:
        return events

    # 1. 创建一个副本，以避免修改原始数据
    noisy_events = events.copy()
    
    # 2. 确定要修改的事件数量和索引
    num_events = len(events)
    num_to_modify = int(num_events * percentage)

    if num_to_modify == 0:
        # print("Warning: 0 events selected to modify.")
        return noisy_events

    # 随机选择 'num_to_modify' 个不重复的索引
    indices_to_modify = np.random.choice(num_events, size=num_to_modify, replace=False)

    # 3. 获取这些索引对应的极性列
    original_p = noisy_events[indices_to_modify, 3]

    # 4. 反转极性 (假设为 0/1 编码)
    # 0 变为 1 (1 - 0)
    # 1 变为 0 (1 - 1)
    flipped_p = 1 - original_p
    
    # --- 备选方案：如果您的编码是 -1/+1 ---
    # flipped_p = original_p * -1
    # ------------------------------------

    # 5. 将修改后的极性放回数组中
    noisy_events[indices_to_modify, 3] = flipped_p

    return noisy_events


def add_thermal_noise(
    events: np.ndarray, 
    hw: tuple[int, int], 
    mean_noise_rate: float = 1.0, 
    std_noise_rate: float = 2.0,
    time_scale_to_seconds: float = 1e6
) -> np.ndarray:
    """
    向DVS事件流中添加模拟的热噪声。
    （已更新，修复了时间单位问题导致的内存爆炸）

    该函数模拟了每个像素的独立泊松噪声源，允许
    噪声率在像素之间变化（模拟“热像素”）。

    参数:
    events (np.ndarray):
        输入的事件数组，形状为 (N, 4)，列为 (x, y, t, p)。
    
    hw (tuple[int, int]):
        传感器的高度和宽度 (H, W)。
    
    mean_noise_rate (float):
        所有像素的平均噪声率（单位：Hz，即 事件数/秒/像素）。
    
    std_noise_rate (float):
        噪声率在像素间的标准差（单位：Hz）。
        - 如果为 0，则所有像素具有完全相同的 `mean_noise_rate`。
        - 如果大于 0，则每个像素的噪声率将从**对数正态分布**
          中抽取，该分布的均值为 `mean_noise_rate`，
          标准差为 `std_noise_rate`。
    
    time_scale_to_seconds (float):
        用于将事件时间戳 't' 转换为秒的换算系数。
        - 默认: 1e6 (假设 't' 的单位是 微秒, µs)
        - 如果 't' 是 纳秒 (ns), 使用 1e9
        - 如果 't' 已经是 秒 (s), 使用 1.0

    返回:
    np.ndarray:
        一个新的事件数组，形状为 (N + M, 4)，包含了原始事件和
        新生成的噪声事件，并已按时间戳 (t) 排序。
    """
    
    height, width = hw
    
    # --- 1. 确定模拟的总时长 T_max (以秒为单位) ---
    if events.shape[0] > 0:
        T_max_orig_units = events[:, 2].max()
    else:
        # 如果没有输入事件，我们无法确定T_max，因此不添加噪声。
        return events

    # 将 T_max 转换为秒
    T_max_seconds = T_max_orig_units / time_scale_to_seconds
    
    if T_max_seconds <= 0:
        # 持续时间为0或负，不添加噪声
        return events

    # --- 2. 为每个像素生成噪声率 (H, W) ---
    # (使用对数正态分布 (Lognormal) 模拟)
    
    if mean_noise_rate <= 0:
        pixel_rates = np.zeros((height, width))
    
    elif std_noise_rate == 0:
        pixel_rates = np.full((height, width), mean_noise_rate)
        
    else:
        M = mean_noise_rate
        S = std_noise_rate
        
        sigma_sq = np.log((S / M)**2 + 1)
        mu = np.log(M) - sigma_sq / 2
        sigma = np.sqrt(sigma_sq) 
        
        pixel_rates = np.random.lognormal(
            mean=mu,
            sigma=sigma,
            size=(height, width)
        )

    # --- 3. 计算聚合噪声并生成噪声事件总数 ---
    # 整个传感器的总噪声率（所有像素率的总和, 单位: events/sec）
    total_rate = np.sum(pixel_rates)
    
    if total_rate <= 0:
        return events

    # 预期噪声事件的总数 (现在单位正确了)
    # (events/sec) * (sec)
    expected_num_noise_events = total_rate * T_max_seconds
    
    # 噪声事件的实际数量是从泊松分布中抽取的
    num_noise_events = np.random.poisson(expected_num_noise_events)
    
    if num_noise_events == 0:
        return events

    # --- 4. 矢量化生成所有噪声事件的属性 ---
    
    # (a) 时间戳 (t):
    # 1. 在 [0, T_max_seconds] 范围内生成 (单位: 秒)
    noise_t_seconds = np.random.uniform(0, T_max_seconds, size=num_noise_events)
    # 2. 转换回原始时间单位 (例如 µs)
    noise_t = noise_t_seconds * time_scale_to_seconds
    
    # (b) 极性 (p):
    noise_p = np.random.choice([0, 1], size=num_noise_events)
    
    # (c) 位置 (x, y):
    rates_flat = pixel_rates.flatten()
    # 归一化概率分布 (避免除以0)
    total_rate_safe = np.sum(rates_flat)
    if total_rate_safe == 0: # 极端情况，虽然前面检查过了
        return events
    prob_dist = rates_flat / total_rate_safe
    
    pixel_indices = np.random.choice(
        height * width, 
        size=num_noise_events, 
        p=prob_dist
    )
    noise_y = pixel_indices // width
    noise_x = pixel_indices % width

    # --- 5. 组装噪声事件 ---
    # 确保数据类型与原始事件一致
    noise_events = np.stack([
        noise_x.astype(events.dtype),
        noise_y.astype(events.dtype),
        noise_t.astype(events.dtype),
        noise_p.astype(events.dtype)
    ], axis=1)

    # --- 6. 合并与排序 ---
    noisy_events = np.concatenate((events, noise_events), axis=0)
    sort_indices = np.argsort(noisy_events[:, 2])
    
    return noisy_events[sort_indices]



@jit(nopython=True, cache=True)
def _filter_spatio_temporal_numba(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    timestamps: np.ndarray,
    sae: np.ndarray,         # (H, W) array, modified in-place
    keep_mask: np.ndarray,  # (N,) bool array, modified in-place
    height: int,
    width: int,
    spatial_radius: int,
    temporal_window: float,
    min_supporters: int
):
    """
    Numba-accelerated core loop for spatio-temporal filtering.
    Modifies sae and keep_mask in place.
    (这是Numba JIT编译的核心函数)
    """
    num_events = timestamps.shape[0]
    
    for i in range(num_events):
        x = x_coords[i]
        y = y_coords[i]
        t = timestamps[i]

        # 定义空间邻域的边界 (Numba JIT 兼容)
        x_min = max(0, x - spatial_radius)
        x_max = min(width - 1, x + spatial_radius)
        y_min = max(0, y - spatial_radius)
        y_max = min(height - 1, y + spatial_radius)

        # 使用嵌套循环计算支持者 (对Numba最友好的方式)
        num_supporters = 0
        # +1 是因为切片是包含右边界的
        for y_n in range(y_min, y_max + 1):
            for x_n in range(x_min, x_max + 1):
                if sae[y_n, x_n] >= t - temporal_window:
                    num_supporters += 1

        # 应用过滤规则
        if num_supporters >= min_supporters:
            keep_mask[i] = True
        
        # 无论如何，都用当前事件更新SAE
        sae[y, x] = t

def filter_spatio_temporal(
    events: np.ndarray, 
    hw: tuple[int, int], 
    spatial_radius: int = 1, 
    temporal_window: float = 5000.0, 
    min_supporters: int = 2
) -> np.ndarray:
    """
    使用 高效的、Numba JIT 加速的 “事件平面”（SAE）算法，对DVS事件流进行时空去噪。
    
    这是一个 O(N) 算法，N 是事件的数量。
    它假定事件已按时间 (t) 排序。

    参数:
    events (np.ndarray):
        输入的事件数组，形状为 (N, 4)，列为 (x, y, t, p)。
        必须已按 t 排序！
    
    hw (tuple[int, int]):
        传感器的高度和宽度 (H, W)。
    
    spatial_radius (int):
        空间邻域半径。
        - r=0: 1x1 邻域 (仅自己)
        - r=1: 3x3 邻域
        - r=2: 5x5 邻域
    
    temporal_window (float):
        时间窗口 (delta_t)。一个事件要成为“支持者”，
        其时间戳必须在 [t - delta_t, t] 范围内。
        (注意：单位应与事件 't' 的单位一致，例如 µs)

    min_supporters (int):
        在时空圆柱体中保留一个事件所需的最小事件数
        （包括事件本身）。
        - 1: 保留所有事件 (无过滤)
        - 2: 至少有1个邻居（或自己）在时空窗口内
        - 3: 至少有2个邻居（或自己）在时空窗口内

    返回:
    np.ndarray:
        一个新的事件数组，形状为 (M, 4) (M <= N)，
        只包含通过过滤的事件。
    """
    
    height, width = hw
    
    # 检查事件是否为空
    if events.shape[0] == 0:
        return events
        
    # 1. 创建“事件平面”(Surface of Active Events, SAE)
    # (这部分在Numpy中已经足够快)
    sae_init_time = events[0, 2] - temporal_window - 1.0
    sae = np.full(hw, sae_init_time, dtype=np.float64)
    
    # 2. 创建一个布尔掩码，用于标记要保留的事件
    keep_mask = np.zeros(events.shape[0], dtype=bool)
    
    # 3. 提取坐标和时间戳
    # (这部分在Numpy中已经足够快)
    x_coords = events[:, 0].astype(int)
    y_coords = events[:, 1].astype(int)
    timestamps = events[:, 2]

    # 4. 调用 Numba JIT 加速的核心函数
    _filter_spatio_temporal_numba(
        x_coords,
        y_coords,
        timestamps,
        sae,          # 传入以在原地修改
        keep_mask,    # 传入以在原地修改
        height,
        width,
        spatial_radius,
        temporal_window,
        min_supporters
    )

    # 10. 返回被标记为保留的事件
    return events[keep_mask]


try:
    import faiss
except ImportError:
    faiss = None



# ---------------------------------------------

def k_mean_cluster_faiss_gpu(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                           sample_number: int, H: int, W: int, scale_t: float, WORKER_FAISS_RES, niter=20):
    """
    使用 Faiss GPU 对时空点 (x, y, t) 进行 K-Means 聚类。
    此版本设计为在 Dataloader worker 中运行, 依赖 worker_init_fn 设置全局资源。
    """
    
    # 检查资源是否已在 worker 中初始化

    n = x.size

    # 如果点数已经小于等于目标采样数，则无需聚类，直接返回
    if n <= sample_number:
        return x, y, t, p, np.ones_like(x)

    # 1. 数据归一化
    t = t.astype(float) - t[0]
    t_last = t[-1]
    if t_last == 0:
        t_last = 1.0  # 避免除以零

    points = np.column_stack((
        x.astype(float) / (W - 1),
        y.astype(float) / (H - 1),
        t / t_last * scale_t
    ))
    
    # Faiss 需要 float32 类型
    points = points.astype('float32')
    
    d = points.shape[1]  # 数据维度 (应为 3)

    # 2. 根据极性 p 拆分数据
    p_bool = p.astype(bool)
    points_1 = points[p_bool]
    points_0 = points[~p_bool]

    # 3. 计算两组各自的聚类数
    sample_number_1 = int(p_bool.mean() * sample_number)
    sample_number_0 = sample_number - sample_number_1

    # 4. 初始化 Faiss GPU 资源 (移除 - 使用全局)
    
    # --- 5. 对 Group 1 (p=True) 进行聚类 ---
    
    # 检查边缘情况
    if points_1.shape[0] == 0 or sample_number_1 == 0:
        intensity_1 = np.zeros(sample_number_1, dtype='int64')
        centroids_1 = np.zeros((sample_number_1, d), dtype='float32')
    elif points_1.shape[0] < sample_number_1:
        # 点数少于聚类数，将所有点作为质心，并用0填充
        print(f"Warning: Group 1 has {points_1.shape[0]} points, but {sample_number_1} clusters requested. Using all points.")
        centroids_1 = np.zeros((sample_number_1, d), dtype='float32')
        centroids_1[:points_1.shape[0], :] = points_1
        
        intensity_1 = np.zeros(sample_number_1, dtype='int64')
        intensity_1[:points_1.shape[0]] = 1 # 每个点自成一簇
    else:
        # (1) 训练 K-Means 获取质心
        kmeans_1 = faiss.Kmeans(d=d, k=sample_number_1, niter=niter, nredo=5, gpu=0, seed=0, min_points_per_centroid=1, max_points_per_centroid=10000000)
        kmeans_1.train(points_1)
        centroids_1 = kmeans_1.centroids

        # (2) 分配标签以计算 intensity
        # 修复: 不再直接调用 GpuIndexFlatL2 构造函数
        # 而是先创建 CPU 索引，再克隆到指定的 GPU
        cpu_index_1 = faiss.IndexFlatL2(d)
        index_1 = faiss.index_cpu_to_gpu(WORKER_FAISS_RES, 0, cpu_index_1)
        
        index_1.add(centroids_1)
        # 搜索每个点最近的质心 (k=1)
        D_1, I_1 = index_1.search(points_1, 1)
        labels_1 = I_1.ravel()
        intensity_1 = np.bincount(labels_1, minlength=sample_number_1)

    # 反归一化
    x_1 = centroids_1[:, 0] * (W - 1)
    y_1 = centroids_1[:, 1] * (H - 1)
    t_1 = centroids_1[:, 2] / scale_t


    # --- 6. 对 Group 0 (p=False) 进行聚类 ---
    
    # 检查边缘情况
    if points_0.shape[0] == 0 or sample_number_0 == 0:
        intensity_0 = np.zeros(sample_number_0, dtype='int64')
        centroids_0 = np.zeros((sample_number_0, d), dtype='float32')
    elif points_0.shape[0] < sample_number_0:
        # 点数少于聚类数
        print(f"Warning: Group 0 has {points_0.shape[0]} points, but {sample_number_0} clusters requested. Using all points.")
        centroids_0 = np.zeros((sample_number_0, d), dtype='float32')
        centroids_0[:points_0.shape[0], :] = points_0
        
        intensity_0 = np.zeros(sample_number_0, dtype='int64')
        intensity_0[:points_0.shape[0]] = 1
    else:
        # (1) 训练 K-Means 获取质心
        kmeans_0 = faiss.Kmeans(d=d, k=sample_number_0, niter=niter, nredo=5, gpu=0, seed=0, min_points_per_centroid=1, max_points_per_centroid=10000000)
        kmeans_0.train(points_0)
        centroids_0 = kmeans_0.centroids

        # (2) 分配标签以计算 intensity
        # 修复: 不再直接调用 GpuIndexFlatL2 构造函数
        cpu_index_0 = faiss.IndexFlatL2(d)
        index_0 = faiss.index_cpu_to_gpu(WORKER_FAISS_RES, 0, cpu_index_0)
        
        index_0.add(centroids_0)
        D_0, I_0 = index_0.search(points_0, 1)
        labels_0 = I_0.ravel()
        intensity_0 = np.bincount(labels_0, minlength=sample_number_0)

    # 反归一化
    x_0 = centroids_0[:, 0] * (W - 1)
    y_0 = centroids_0[:, 1] * (H - 1)
    t_0 = centroids_0[:, 2] / scale_t

    
    # --- 7. 合并与排序 ---
    
    intensity = np.concatenate((intensity_1, intensity_0))
    x = np.concatenate((x_1, x_0))
    y = np.concatenate((y_1, y_0))
    t = np.concatenate((t_1, t_0))
    
    # 创建新的极性数组
    p_new = np.zeros(sample_number, dtype=bool)
    p_new[0: sample_number_1] = True

    # 根据时间 t 排序
    indices = np.argsort(t)
    x, y, t, p_new, intensity = x[indices], y[indices], t[indices], p_new[indices], intensity[indices]

    return x, y, t, p_new, intensity


def init_faiss():

    WORKER_FAISS_RES = faiss.StandardGpuResources()
    return WORKER_FAISS_RES
        


class EventNPDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, training: bool, root: str, sample_number: int, sampler: str,
                 repeats: int = 1, transform=None):
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

        # self.WORKER_FAISS_RES = None

        self.transform = transform

    def read_npz_by_key(self, opened_npz, k: str):
        try:
            ret = opened_npz[k]
        except BaseException:
            ret = None
        return ret

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

        sample = np.load(path, mmap_mode=None if self.sampler is None else 'c')

        t = self.read_npz_by_key(sample, 't')
        y = self.read_npz_by_key(sample, 'y')
        x = self.read_npz_by_key(sample, 'x')
        p = self.read_npz_by_key(sample, 'p')
        n = t.shape[0]
        intensity = None

        # events = np.stack((x, y, t, p), axis=1)
        # events = add_thermal_noise(events, (128, 128), mean_noise_rate=0.005, std_noise_rate=1)
        
        # events = filter_spatio_temporal(events, (128, 128), spatial_radius=3, temporal_window=5000, min_supporters=3)
        # x = events[:, 0]
        # y = events[:, 1]
        # t = events[:, 2]
        # p = events[:, 3]

        
        if self.sampler is not None:
            x, y, t, p = self.sampler(self.rng, x, y, t, p, self.sample_number)

        else:
            intensity = self.read_npz_by_key(sample, 'intensity')


        
        # x, y, t, p, intensity = k_mean_cluster_faiss_gpu(x=x, y=y, t=t, p=p, sample_number=1024, H=128, W=128, scale_t=1, WORKER_FAISS_RES=self.WORKER_FAISS_RES, niter=100)
        # x, y, t, p, intensity = k_mean_cluster(x=x, y=y, t=t, p=p, sample_number=1024, H=128, W=128, scale_t=1, mini=False)



        t = np.ascontiguousarray(t)
        t -= t.flat[0]
        t = torch.from_numpy(t).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        p = torch.from_numpy(np.ascontiguousarray(p)).float()


        if intensity is not None:
            intensity = torch.from_numpy(np.ascontiguousarray(intensity)).float()


        if self.transform is not None:
            x, y, t, p, intensity = self.transform(x, y, t, p, intensity, path, label)


        # 使用字典作为返回值 更直观

        rets = {'x': x, 'y': y, 't': t, 'p': p, 'label': label}

        if intensity is not None:
            rets['intensity'] = intensity
        
        if self.repeats > 1 and not self.training:
            # 测试时用来计算acc std
            rets['indices'] = i

        return rets


def event_collate_fun_with_padding(batch: list):

    keys = tuple(batch[0].keys())

    batched = {}

    for key in keys:
        batched[key] = []

    for item in batch:
        for key in keys:
            batched[key].append(item[key])

    for key in keys:
        if isinstance(batched[key][0], torch.Tensor):
            batched[key] = pad_sequence(batched[key], batch_first=True, padding_value=-1)

        else:
            batched[key] = torch.as_tensor(batched[key])
    
    batched['valid_mask'] = batched['p'] >= 0
    return batched
import torch.distributed as dist
class MPerClassSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, labels, m_per_class, batch_size):
        self.labels = np.array(labels)
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        
        # --- DDP 多卡处理逻辑 ---
        # 尝试获取 DDP 信息，如果没开启 DDP，默认当作单卡
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size() # 总卡数
            self.rank = dist.get_rank()               # 当前卡 ID
        else:
            self.num_replicas = 1
            self.rank = 0
            
        self.epoch = 0 # 用于设置随机种子
        
        # 1. 整理数据索引
        self.classes = np.unique(self.labels)
        self.indices_by_class = {}
        for c in self.classes:
            self.indices_by_class[c] = np.where(self.labels == c)[0]
            
        self.classes_per_batch = self.batch_size // self.m_per_class
        self.n_samples = len(self.labels)
        
        # --- 关键修改：计算每个 Epoch 的 Batch 数量 ---
        # 总 Batch 数需要除以卡数，否则每个 Epoch 会遍历 world_size 遍数据
        self.n_batches = (self.n_samples // self.batch_size) // self.num_replicas

    def __iter__(self):
        # --- 关键修改：确保每张卡的随机种子不同 ---
        # 种子 = 基础种子 + 当前 Epoch + 当前卡 ID
        # 这样保证：
        # 1. 每张卡采样的组合不同 (通过 rank)
        # 2. 每个 Epoch 采样的组合不同 (通过 epoch)
        np.random.seed(42 + self.epoch + self.rank)
        
        for _ in range(self.n_batches):
            batch_indices = []
            selected_classes = np.random.choice(
                self.classes, 
                self.classes_per_batch, 
                replace=(len(self.classes) < self.classes_per_batch)
            )
            
            for c in selected_classes:
                indices = self.indices_by_class[c]
                selected_indices = np.random.choice(
                    indices, 
                    self.m_per_class, 
                    replace=(len(indices) < self.m_per_class)
                )
                batch_indices.extend(selected_indices)
            
            yield batch_indices

    def __len__(self):
        return self.n_batches

    def set_epoch(self, epoch):
        # Lightning 会自动调用这个方法
        self.epoch = epoch
    
if __name__ == '__main__':
    pass