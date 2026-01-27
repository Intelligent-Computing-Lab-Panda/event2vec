import torch
import os
import torchvision
import numpy as np
import torch.nn.functional as F
import tqdm
import lightning
from sklearn.cluster import MiniBatchKMeans, KMeans
import time
lightning.seed_everything(0)


def k_mean_cluster_sklearn(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                  sample_number: int, H: int, W: int, scale_t: float, mini:bool=False, mini_batch_size:int=1024, max_iter:int=300):
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
        kmeans = MiniBatchKMeans(n_clusters=sample_number_1, random_state=0, n_init='auto', batch_size=mini_batch_size, max_iter=max_iter)
    else:
        kmeans = KMeans(n_clusters=sample_number_1, random_state=0, n_init='auto', max_iter=max_iter)
    kmeans.fit(points_1)
    intensity = np.bincount(kmeans.labels_, minlength=sample_number_1)
    x = kmeans.cluster_centers_[:, 0] * (W - 1)
    y = kmeans.cluster_centers_[:, 1] * (H - 1)
    t = kmeans.cluster_centers_[:, 2] / scale_t




    if mini:
        kmeans = MiniBatchKMeans(n_clusters=sample_number_0, random_state=0, n_init='auto', batch_size=mini_batch_size, max_iter=max_iter)
    else:
        kmeans = KMeans(n_clusters=sample_number_0, random_state=0, n_init='auto', max_iter=max_iter)

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


def k_mean_cluster(x, y, t, p, sample_number, H, W, scale_t=1.0, max_iter=20, tol=1e-4, batch_size=64):
    """
    Batched K-Means++ on GPU
    :param batch_size: 每次迭代采样的中心数量。
                       32/64 是很好的平衡点。
                       越大越快，但理论上对 K-Means++ 的分布破坏越大（不过实测影响微乎其微）。
    """
    device = torch.device('cuda')
    
    # 1. 数据上传与预处理
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device, non_blocking=True).float()
        y = torch.from_numpy(y).to(device, non_blocking=True).float()
        t = torch.from_numpy(t).to(device, non_blocking=True).float()
        p = torch.from_numpy(p).to(device, non_blocking=True).float()

    n = x.shape[0]
    if n <= sample_number:
        return x, y, t, p, torch.ones_like(x)

    # 归一化
    t_span = t[-1] - t[0]
    if t_span < 1e-6: t_span = 1.0
    t_norm = (t - t[0]) / t_span * scale_t
    
    points = torch.stack((
        x / (W - 1),
        y / (H - 1),
        t_norm
    ), dim=1) # [N, 3]

    p_bool = p.bool()
    
    # 极性计数与分配
    p_float = p.float()
    cnt_1 = int(p_float.sum().item())
    
    sample_number_1 = int(p_float.mean().item() * sample_number)
    sample_number_1 = max(1, min(sample_number_1, cnt_1))
    sample_number_0 = max(1, min(sample_number - sample_number_1, n - cnt_1))

    results = []
    configs = [(p_bool, sample_number_1), (~p_bool, sample_number_0)]
    
    for mask, k in configs:
        pts = points[mask]
        M = pts.shape[0]
        if M == 0: continue
        
        # ======================================================
        # Phase 1: Batched K-Means++ Initialization (核心改进)
        # ======================================================
        
        # 1.1 随机选第一个点
        centers = torch.empty((k, 3), device=device, dtype=pts.dtype)
        first_idx = torch.randint(0, M, (1,), device=device)
        centers[0] = pts[first_idx]
        
        # 1.2 维护每个点到最近中心的距离平方 (min_dist^2)
        # 初始化为到第一个点的距离
        closest_dist_sq = torch.sum((pts - centers[0]) ** 2, dim=1)
        
        current_count = 1
        
        # 1.3 批量循环采样
        # 循环次数 = K / batch_size (约 16-30 次，速度很快)
        while current_count < k:
            # 本轮需要选多少个
            needed = k - current_count
            this_batch = min(needed, batch_size)
            
            # 按距离平方作为概率权重进行采样
            # 加上 epsilon 防止全 0
            weights = closest_dist_sq + 1e-10
            
            # 核心：一次选出 this_batch 个候选点
            candidate_indices = torch.multinomial(weights, this_batch, replacement=False)
            new_centers_batch = pts[candidate_indices]
            
            # 填入中心 tensor
            centers[current_count : current_count + this_batch] = new_centers_batch
            
            # 更新最短距离 (关键优化：只计算点到“新加入中心”的距离，然后和老的 min 比较)
            # new_dists: [M, this_batch]
            new_dists = torch.cdist(pts, new_centers_batch).pow(2)
            # new_min: [M]
            new_min, _ = torch.min(new_dists, dim=1)
            # 更新全局 min
            closest_dist_sq = torch.min(closest_dist_sq, new_min)
            
            current_count += this_batch

        old_centers = centers.clone()
        
        # ======================================================
        # Phase 2: Standard Lloyd's Iteration
        # ======================================================
        # 由于初始化极好，这里的迭代通常 5-10 次内就收敛
        for i in range(max_iter):
            dists = torch.cdist(pts, centers)
            labels = torch.argmin(dists, dim=1)
            
            # 快速计算新中心
            counts = torch.bincount(labels, minlength=k).float()
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, 3), pts)
            
            # 处理空簇
            mask_empty = counts == 0
            counts[mask_empty] = 1.0
            new_centers = new_centers / counts.unsqueeze(1)
            
            if mask_empty.any():
                new_centers[mask_empty] = old_centers[mask_empty]
            
            centers = new_centers
            
            shift = torch.norm(centers - old_centers, dim=1).mean()
            if shift < tol:
                break
            old_centers = centers.clone()
            
        intensity = torch.bincount(labels, minlength=k).float()
        
        cx = centers[:, 0] * (W - 1)
        cy = centers[:, 1] * (H - 1)
        ct = centers[:, 2] / scale_t * t_span + t[0]
        
        results.append((cx, cy, ct, intensity))

    # 合并
    if not results: return x, y, t, p, torch.ones_like(x)

    cat_x = torch.cat([r[0] for r in results])
    cat_y = torch.cat([r[1] for r in results])
    cat_t = torch.cat([r[2] for r in results])
    cat_intensity = torch.cat([r[3] for r in results])
    
    cat_p = torch.zeros(cat_x.shape[0], device=device, dtype=torch.float)
    if len(results) >= 1 and sample_number_1 > 0:
         cat_p[:results[0][0].shape[0]] = 1.0

    sort_idx = torch.argsort(cat_t)
    
    # 确保 Contiguous，这对后续模型推理的内存访问非常重要
    return (
        cat_x[sort_idx].contiguous(), 
        cat_y[sort_idx].contiguous(), 
        cat_t[sort_idx].contiguous(), 
        cat_p[sort_idx].contiguous(), 
        cat_intensity[sort_idx].contiguous()
    )



try:
    import faiss
    def run_faiss_kmeans(data: np.ndarray, n_clusters: int, n_iter: int, gpu_id: int, res: faiss.StandardGpuResources):
        """
        使用 faiss-gpu 运行 K-Means 聚类的辅助函数。
        """
        if data.shape[0] == 0 or n_clusters == 0:
            # 返回正确形状的空数组
            d = data.shape[1] if data.ndim > 1 else 3 # 假设 3D (x,y,t)
            return np.zeros((n_clusters, d), dtype=np.float32), np.zeros(n_clusters, dtype=np.int32)
            
        n_samples, d = data.shape
        
        if n_samples < n_clusters:
            centroids = data
            intensity = np.ones(n_samples, dtype=np.int32)
            centroids_padded = np.zeros((n_clusters, d), dtype=np.float32)
            centroids_padded[:n_samples] = centroids
            intensity_padded = np.zeros(n_clusters, dtype=np.int32)
            intensity_padded[:n_samples] = intensity
            return centroids_padded, intensity_padded

        data = np.ascontiguousarray(data, dtype=np.float32)

        try:
            clus = faiss.Clustering(d, n_clusters)
            clus.niter = n_iter
            clus.verbose = False

            cpu_index = faiss.IndexFlatL2(d)
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

            clus.train(data, gpu_index)

            centroids = faiss.vector_to_array(clus.centroids).reshape(n_clusters, d)

            gpu_index.reset()
            gpu_index.add(centroids)
            _, labels = gpu_index.search(data, 1)
            labels = labels.ravel()
            
            intensity = np.bincount(labels, minlength=n_clusters)
            
            del gpu_index
            del cpu_index
            
            return centroids, intensity.astype(np.int32)

        except RuntimeError as e:
            print(f"FAISS K-Means failed on GPU {gpu_id} with data shape {data.shape} and n_clusters {n_clusters}. Error: {e}", flush=True)
            return np.zeros((n_clusters, d), dtype=np.float32), np.zeros(n_clusters, dtype=np.int32)


    def k_mean_cluster_faiss(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray,
                            sample_number: int, H: int, W: int, scale_t: float, n_iter: int, gpu_id: int, res: faiss.StandardGpuResources):
        """
        使用 faiss-gpu 进行超快速事件采样。
        注意：此函数假定 n > sample_number。
        """
        n = x.size
        # 归一化
        t_start = t[0] if n > 0 else 0
        t_end = t[-1] if n > 0 else 0
        t_duration = t_end - t_start
        t_normalized = (t - t_start) / t_duration if t_duration > 0 else np.zeros_like(t, dtype=np.float32)

        points = np.column_stack((
            x.astype(np.float32) / (W - 1),
            y.astype(np.float32) / (H - 1),
            t_normalized.astype(np.float32) * scale_t
        ))
        
        p_bool = p.astype(bool)
        points_1 = points[p_bool]
        points_0 = points[~p_bool]

        p_mean = p_bool.mean() if n > 0 else 0.5
        sample_number_1 = int(p_mean * sample_number)
        sample_number_0 = sample_number - sample_number_1
        
        # 确保总数正确
        if sample_number_1 + sample_number_0 != sample_number:
            sample_number_1 = sample_number - sample_number_0

        final_x = np.zeros(sample_number, dtype=np.float32)
        final_y = np.zeros(sample_number, dtype=np.float32)
        final_t = np.zeros(sample_number, dtype=np.float32)
        final_intensity = np.zeros(sample_number, dtype=np.int32)
        final_p = np.zeros(sample_number, dtype=bool)
        final_p[0:sample_number_1] = True
        
        # 运行 K-Means
        centers_1, intensity_1 = run_faiss_kmeans(points_1, sample_number_1, n_iter, gpu_id, res)
        if centers_1.shape[0] > 0:
            # centers_1 保证有 sample_number_1 个
            valid_clusters = sample_number_1
            final_intensity[:valid_clusters] = intensity_1
            final_x[:valid_clusters] = centers_1[:, 0] * (W - 1)
            final_y[:valid_clusters] = centers_1[:, 1] * (H - 1)
            final_t[:valid_clusters] = (centers_1[:, 2] / scale_t * t_duration) + t_start

        centers_0, intensity_0 = run_faiss_kmeans(points_0, sample_number_0, n_iter, gpu_id, res)
        if centers_0.shape[0] > 0:
            valid_clusters = sample_number_0
            start_idx = sample_number_1
            end_idx = sample_number_1 + valid_clusters
            final_intensity[start_idx:end_idx] = intensity_0
            final_x[start_idx:end_idx] = centers_0[:, 0] * (W - 1)
            final_y[start_idx:end_idx] = centers_0[:, 1] * (H - 1)
            final_t[start_idx:end_idx] = (centers_0[:, 2] / scale_t * t_duration) + t_start

        # 按时间排序
        indices = np.argsort(final_t)
        return final_x[indices], final_y[indices], final_t[indices], final_p[indices], final_intensity[indices]
except ImportError:
    faiss = None
    print('faiss is not installed')



class EventsNpFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, train: bool, root: str, out_dir: str | None, cluster:str = 'batched'):
        root = os.path.join(root, 'train' if train else 'test')
        if out_dir is not None:
            out_dir = os.path.join(out_dir, 'train' if train else 'test')


        super().__init__(root=root,
                    loader=None,
                    extensions=('npz', 'npy'),
                    transform=None,
                    target_transform=None,
                    is_valid_file=None,
                    allow_empty=False)
        self.train = train

        self.out_dir = out_dir
        self.cluster = cluster
        if cluster == 'faiss':
            self.faiss_res = faiss.StandardGpuResources()
    
    def __len__(self) -> int:

        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]

        sample = np.load(path)
        t = sample['t'].astype(float)
        
        y = sample['y'].astype(float)
        x = sample['x'].astype(float)
        p = sample['p'].astype(bool)

        if self.cluster == 'batched':
            x, y, t, p, intensity = k_mean_cluster(x, y, t, p, sample_number=1024, H=128, W=128, scale_t=1)
        elif self.cluster == 'sklearn':
            x, y, t, p, intensity = k_mean_cluster_sklearn(x, y, t, p, sample_number=1024, H=128, W=128, scale_t=1, max_iter=20)
        elif self.cluster == 'faiss':
            x, y, t, p, intensity = k_mean_cluster_faiss(x, y, t, p, sample_number=1024, H=128, W=128, scale_t=1, gpu_id=0, n_iter=300, res=self.faiss_res)


        if self.out_dir is not None:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
                y = y.cpu().numpy()
                t = t.cpu().numpy()
                p = p.cpu().numpy()
                intensity = intensity.cpu().numpy()
            label_dir = os.path.join(self.out_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            fname = os.path.join(label_dir, os.path.basename(path).split('.')[0] + '.npz')
            np.savez(fname, x=x, y=y, t=t, p=p, intensity=intensity)
        
        


        # return x, y, t, p, intensity, label
        return 0

if __name__ == '__main__':
    benchmark = False
    cluster = 'batched'
    if benchmark:
        out_dir = None
    else:
        out_dir = f'/dev/shm/dvs_lip/kmean_1024'

    for train in (False, True):
        dts = EventsNpFolder(train=train, root='/dev/shm/dvs_lip', out_dir=out_dir, cluster=cluster)


        if benchmark:
            torch.cuda.synchronize()
            ts = [time.perf_counter()]
            for item in tqdm.tqdm(dts):
                torch.cuda.synchronize()
                ts.append(time.perf_counter())
            ts = torch.as_tensor(ts).diff()
            mean = torch.mean(ts).item() * 1000
            std = torch.std(ts).item() * 1000
            print(f'{cluster} speed = {mean} ± {std} ms')
            break
        else:
            loader = torch.utils.data.DataLoader(dataset=dts, batch_size=8, num_workers=8)

            for item in tqdm.tqdm(loader):
                pass
'''
batched speed = 17.097989097237587 ± 15.556978061795235 ms
sklearn 
    iters=300 speed = 383.4170997142792 ± 149.22188222408295 ms


python train_script.py --config ./config/dvs_lip_cluster_event_self_supervised_training.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.mask_len 10 --data.root /dev/shm/dvs_lip/kmean_1024_sklearn_300

python train_script.py --config ./config/dvs_lip_cluster_event.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.train_transform_args RandomResize-0.8,1.2-0.8,1.2/RandomRotation-128,128-15/RandomShear-0.05,0.05/RandomHorizontalFlip-128-1/RandomTranslate-16,16/RandomErasing-128,128-0.1-16,16/RandomChunkDrop-4,128 --data.root /dev/shm/dvs_lip/kmean_1024_sklearn_300 --model.load ./dvslip/checkpoints/version_36/last.ckpt

valid_loss=1.683956, valid_acc=0.750804, valid_acc_std= 0.000000, valid_speed=3011.483382 msec



sklearn 
    iters=20 sklearn speed = 374.2211163043976 ± 145.96940577030182 ms
    
python train_script.py --config ./config/dvs_lip_cluster_event_self_supervised_training.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.mask_len 10 --data.root /dev/shm/dvs_lip/kmean_1024_sklearn_20

python train_script.py --config ./config/dvs_lip_cluster_event.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.train_transform_args RandomResize-0.8,1.2-0.8,1.2/RandomRotation-128,128-15/RandomShear-0.05,0.05/RandomHorizontalFlip-128-1/RandomTranslate-16,16/RandomErasing-128,128-0.1-16,16/RandomChunkDrop-4,128 --data.root /dev/shm/dvs_lip/kmean_1024_sklearn_20 --model.load ./dvslip/checkpoints/version_34/last.ckpt

valid_loss=1.680527, valid_acc=0.747186, valid_acc_std= 0.000000, valid_speed=2992.797463 msec


1 iters=20 faiss speed = 15.948493033647537 ± 17.586009576916695 ms

python train_script.py --config ./config/dvs_lip_cluster_event_self_supervised_training.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.mask_len 10 --data.root /dev/shm/dvs_lip/kmean_1024_faiss_20

python train_script.py --config ./config/dvs_lip_cluster_event.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.train_transform_args RandomResize-0.8,1.2-0.8,1.2/RandomRotation-128,128-15/RandomShear-0.05,0.05/RandomHorizontalFlip-128-1/RandomTranslate-16,16/RandomErasing-128,128-0.1-16,16/RandomChunkDrop-4,128 --data.root /dev/shm/dvs_lip/kmean_1024_faiss_20 --model.load ./dvslip/checkpoints/version_30/last.ckpt
valid_loss=1.706005, valid_acc=0.743971, valid_acc_std= 0.000000, valid_speed=3009.935944 msec 


iters=300 faiss speed = 162.41206228733063 ± 126.24216079711914 ms
python train_script.py --config ./config/dvs_lip_cluster_event_self_supervised_training.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.mask_len 10 --data.root /dev/shm/dvs_lip/kmean_1024_faiss_300iters
python train_script.py --config ./config/dvs_lip_cluster_event.yaml --model.intensity_norm log --trainer.devices '[4,5,6,7]' --model.train_transform_args RandomResize-0.8,1.2-0.8,1.2/RandomRotation-128,128-15/RandomShear-0.05,0.05/RandomHorizontalFlip-128-1/RandomTranslate-16,16/RandomErasing-128,128-0.1-16,16/RandomChunkDrop-4,128 --data.root /dev/shm/dvs_lip/kmean_1024_faiss_300iters --model.load ./dvslip/checkpoints/version_32/last.ckpt
valid_loss=1.692444, valid_acc=0.741158, valid_acc_std= 0.000000, valid_speed=3040.874953 msec

'''