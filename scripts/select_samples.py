r"""
数据清洗工具：从大量图像中筛选最具代表性的样本

使用项目已有的 DINOv3 ViT-L16 (safetensors) 权重进行特征提取，
采用 hybrid 混合策略（CoreSet + 聚类中心 + 密度边界 + 随机）筛选代表性图像。

用法:
  python data_clening/select_samples.py --src_dir /path/to/images --n_select 300
  
  # 指定输出目录和采样方法
  python data_clening/select_samples.py --src_dir ./data/good --n_select 200 --method hybrid --dst_dir ./data/selected

  # 跳过可视化（服务器环境）
  python data_clening/select_samples.py --src_dir ./data/good --n_select 300 --no_vis
"""

import os
import sys
import argparse
import shutil
import time
import numpy as np

_project_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_project_dir)
_root_dir = os.path.dirname(_parent_dir)
# RAD_APP must come before RAD_reproduction so local modelscope/ wins
for p in [_project_dir, _parent_dir, _root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Add RAD_reproduction for dinov3 package (append to end to avoid shadowing modelscope)
_data_dir = os.path.join(_root_dir, 'RAD_reproduction')
if _data_dir not in sys.path:
    sys.path.append(_data_dir)

from modelscope.config import IMAGENET_MEAN, IMAGENET_STD, WEIGHT_SEARCH_PATHS, LAYER_CONFIGS, find_weight_path


# ============================================================
# Step 1: 特征提取 — DINOv3 (safetensors)
# ============================================================
def extract_features_dinov3(src_dir, model_name='dinov3_vitl16', device='cuda',
                            img_size=512, batch_size=8):
    """使用 DINOv3 提取多层 patch 特征，融合为单张图像的特征向量"""
    import torch
    from PIL import Image
    from modelscope.loader import load_model, detect_weight_format

    weight_path = find_weight_path(model_name)

    fmt = detect_weight_format(weight_path)
    print(f"加载模型: {model_name} (格式: {fmt})")
    model = load_model(model_name=model_name, weight_path=weight_path)
    model.to(device).eval()

    layer_indices = LAYER_CONFIGS[model_name]
    print(f"提取层: {layer_indices}")

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([
        os.path.join(src_dir, f) for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    print(f"发现 {len(image_files)} 张图像")

    all_features = []
    from torch.utils.data import DataLoader, Dataset

    class ImageList(Dataset):
        def __init__(self, paths, img_size):
            self.paths = paths
            self.img_size = img_size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB').resize(
                (self.img_size, self.img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1).astype(np.float32))
            return tensor

    dataset = ImageList(image_files, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    n_layers = len(layer_indices)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            inter = model.get_intermediate_layers(
                batch, n=layer_indices, return_class_token=True)

            layer_feats = []
            for li, (patch_feat, cls_feat) in enumerate(inter):
                layer_feats.append(cls_feat)
            fused = torch.cat(layer_feats, dim=-1)
            all_features.append(fused.cpu().numpy())

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  特征提取: {min((batch_idx+1)*batch_size, len(image_files))}/{len(image_files)}")

    features = np.concatenate(all_features, axis=0)
    print(f"特征维度: {features.shape}")
    return features, image_files


# ============================================================
# Step 2: 清洗 — 去除可疑异常
# ============================================================
def clean_outliers(features, contamination=0.03):
    """使用 IsolationForest 检测并移除可疑异常样本。contamination=0 时跳过清洗。"""
    if contamination <= 0:
        print(f"  清洗: 跳过 (contamination={contamination})")
        clean_mask = np.ones(len(features), dtype=bool)
        return clean_mask, features.copy()

    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = iso.fit_predict(features_norm)

    clean_mask = labels == 1
    n_outliers = np.sum(labels == -1)
    print(f"  清洗: 移除 {n_outliers} 个可疑样本 ({n_outliers/len(features)*100:.1f}%)")

    return clean_mask, features[clean_mask]


# ============================================================
# Step 3: 采样策略
# ============================================================
def _pca_reduce(features, n_components=50):
    """PCA降维，自动限制维度不超过样本数和特征数"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n_samples, n_feats = features.shape
    max_dim = min(n_components, n_samples, n_feats)
    if max_dim < 1:
        max_dim = 1
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    pca = PCA(n_components=max_dim)
    features_pca = pca.fit_transform(features_norm)
    return features_pca


def kmeans_sampling(features, n_select):
    from sklearn.cluster import KMeans

    n_total = len(features)
    if n_select <= 0 or n_total == 0:
        return np.array([], dtype=int)
    if n_select >= n_total:
        return np.arange(n_total)

    features_pca = _pca_reduce(features, n_components=50)

    kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
    kmeans.fit(features_pca)

    selected_indices = []
    for i in range(n_select):
        cluster_mask = kmeans.labels_ == i
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features_pca[cluster_mask]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        best_local = np.argmin(distances)
        selected_indices.append(cluster_indices[best_local])

    return np.array(selected_indices)


def coreset_sampling(features, n_select):
    n_total = len(features)
    if n_select <= 0 or n_total == 0:
        return np.array([], dtype=int)
    if n_select >= n_total:
        return np.arange(n_total)

    features_pca = _pca_reduce(features, n_components=50)

    n_total = len(features_pca)
    selected = []
    remaining = list(range(n_total))

    global_center = features_pca.mean(axis=0)
    first_idx = np.argmin(np.linalg.norm(features_pca - global_center, axis=1))
    selected.append(first_idx)
    remaining.remove(first_idx)

    min_distances = np.full(n_total, np.inf)
    min_distances[first_idx] = -1

    for step in range(1, n_select):
        last_selected = selected[-1]
        last_feature = features_pca[last_selected]

        for j in remaining:
            dist = np.linalg.norm(features_pca[j] - last_feature)
            min_distances[j] = min(min_distances[j], dist)

        remaining_distances = [(j, min_distances[j]) for j in remaining]
        next_idx = max(remaining_distances, key=lambda x: x[1])[0]

        selected.append(next_idx)
        remaining.remove(next_idx)
        min_distances[next_idx] = -1

        if (step + 1) % 100 == 0:
            print(f"    CoreSet 进度: {step+1}/{n_select}")

    return np.array(selected)


def stratified_density_sampling(features, n_select, n_clusters=20):
    from sklearn.cluster import KMeans
    from collections import Counter

    n_total = len(features)
    if n_select <= 0 or n_total == 0:
        return np.array([], dtype=int)
    if n_select >= n_total:
        return np.arange(n_total)

    n_clusters = min(n_clusters, n_total)

    features_pca = _pca_reduce(features, n_components=50)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_pca)

    cluster_sizes = Counter(labels)
    total = len(features_pca)
    selected_indices = set()

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_features = features_pca[cluster_mask]
        center = kmeans.cluster_centers_[cluster_id]

        n_from = max(1, int(n_select * len(cluster_indices) / total))
        n_from = min(n_from, len(cluster_indices))

        distances = np.linalg.norm(cluster_features - center, axis=1)
        sorted_local = np.argsort(distances)

        if n_from >= len(cluster_indices):
            sampled_local = list(range(len(cluster_indices)))
        else:
            sampled_local = np.linspace(0, len(cluster_indices) - 1, n_from, dtype=int)

        for local_idx in sampled_local:
            selected_indices.add(cluster_indices[sorted_local[local_idx]])

    return np.array(list(selected_indices))[:n_select]


def hybrid_sampling(features, n_select):
    """混合策略: 40% CoreSet + 30% 聚类中心 + 20% 密度边界 + 10% 随机"""
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors

    n_total = len(features)
    if n_select <= 0 or n_total == 0:
        return np.array([], dtype=int)
    n_select = min(n_select, n_total)

    if n_total <= n_select:
        print(f"  样本数({n_total}) <= 目标数({n_select}), 全部保留")
        return np.arange(n_total)

    features_pca = _pca_reduce(features, n_components=50)

    n_total_pca = len(features_pca)
    all_selected = set()

    # 40% CoreSet (ensure >= 1)
    n_coreset = max(1, int(n_select * 0.4))
    coreset_idx = coreset_sampling(features, n_select=n_coreset)
    all_selected.update(coreset_idx)
    print(f"  CoreSet: {len(coreset_idx)} 张")

    # 30% 聚类中心 (ensure >= 1)
    n_cluster = max(1, int(n_select * 0.3))
    n_cluster = min(n_cluster, n_total_pca)
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    kmeans.fit(features_pca)
    for i in range(n_cluster):
        cluster_mask = kmeans.labels_ == i
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_features = features_pca[cluster_mask]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        best = cluster_indices[np.argmin(distances)]
        all_selected.add(best)
    print(f"  +聚类中心: {len(all_selected)} 张")

    # 20% 密度边界 (ensure >= 1)
    n_boundary = max(1, int(n_select * 0.2))
    k_neighbors = min(10, n_total_pca - 1)
    if k_neighbors >= 1:
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nn.fit(features_pca)
        distances_knn, _ = nn.kneighbors(features_pca)
        local_density = distances_knn.mean(axis=1)
    else:
        local_density = np.zeros(n_total_pca)

    remaining = list(set(range(n_total_pca)) - all_selected)
    remaining.sort(key=lambda idx: -local_density[idx])
    for idx in remaining[:n_boundary]:
        all_selected.add(idx)
    print(f"  +密度边界: {len(all_selected)} 张")

    # fill remaining with random
    remaining = list(set(range(n_total_pca)) - all_selected)
    n_random = min(n_select - len(all_selected), len(remaining))
    if n_random > 0:
        random_idx = np.random.choice(remaining, n_random, replace=False)
        all_selected.update(random_idx)
    print(f"  +随机补充: {len(all_selected)} 张")

    return np.array(list(all_selected))[:n_select]


# ============================================================
# Step 4: 采样质量评估
# ============================================================
class SamplingEvaluator:
    def __init__(self, full_features, selected_indices):
        self.full_features = full_features
        self.selected_indices = selected_indices
        self.selected_features = full_features[selected_indices]

    def coverage_score(self):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(self.selected_features)

        unselected_mask = np.ones(len(self.full_features), dtype=bool)
        unselected_mask[self.selected_indices] = False
        unselected = self.full_features[unselected_mask]

        distances, _ = nn.kneighbors(unselected)
        return {
            'avg_distance': float(distances.mean()),
            'max_distance': float(distances.max()),
            'p95_distance': float(np.percentile(distances, 95)),
        }

    def diversity_score(self):
        from sklearn.metrics.pairwise import pairwise_distances
        dist_matrix = pairwise_distances(self.selected_features)
        upper_tri = dist_matrix[np.triu_indices(len(self.selected_features), k=1)]
        return {
            'avg_pairwise_dist': float(upper_tri.mean()),
            'min_pairwise_dist': float(upper_tri.min()),
        }

    def representativeness_score(self):
        full_mean = self.full_features.mean(axis=0)
        full_std = self.full_features.std(axis=0)
        sel_mean = self.selected_features.mean(axis=0)
        sel_std = self.selected_features.std(axis=0)
        return {
            'mean_shift': float(np.linalg.norm(full_mean - sel_mean)),
            'std_shift': float(np.linalg.norm(full_std - sel_std)),
        }

    def full_report(self):
        coverage = self.coverage_score()
        diversity = self.diversity_score()
        represent = self.representativeness_score()
        print(f"  覆盖度  — avg: {coverage['avg_distance']:.4f}, max: {coverage['max_distance']:.4f}, P95: {coverage['p95_distance']:.4f}")
        print(f"  多样性  — avg_pair: {diversity['avg_pairwise_dist']:.4f}, min_pair: {diversity['min_pairwise_dist']:.4f}")
        print(f"  代表性  — mean_shift: {represent['mean_shift']:.4f}, std_shift: {represent['std_shift']:.4f}")


# ============================================================
# 主流程
# ============================================================
def select_representative_samples(src_dir, dst_dir, n_select=300,
                                   method='hybrid', model_name='dinov3_vitl16',
                                   device='cuda', contamination=0.03,
                                   no_vis=False):
    print("=" * 70)
    print("数据清洗：从大量图像中筛选最具代表性的样本")
    print("=" * 70)
    print(f"源目录: {src_dir}")
    print(f"目标目录: {dst_dir}")
    print(f"选择数量: {n_select}")
    print(f"采样方法: {method}")
    print(f"特征提取: {model_name}")
    print(f"清洗比例: {contamination*100:.1f}%")

    # Step 1: 特征提取
    t0 = time.time()
    print(f"\n[Step 1] 特征提取")
    features, image_files = extract_features_dinov3(src_dir, model_name=model_name, device=device)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # Step 2: 清洗
    print(f"\n[Step 2] 数据清洗")
    t1 = time.time()
    clean_mask, clean_features = clean_outliers(features, contamination=contamination)
    clean_files = [image_files[i] for i in range(len(image_files)) if clean_mask[i]]
    print(f"  清洗后: {len(clean_files)} 张")
    print(f"  耗时: {time.time()-t1:.1f}s")

    if len(clean_features) <= n_select:
        print(f"\n清洗后数量 ({len(clean_features)}) <= 目标数量 ({n_select}), 全部保留")
        selected_files = clean_files
    else:
        # Step 3: 采样
        print(f"\n[Step 3] {method} 采样")
        t2 = time.time()

        method_map = {
            'kmeans': kmeans_sampling,
            'coreset': coreset_sampling,
            'stratified': stratified_density_sampling,
            'hybrid': hybrid_sampling,
        }
        sampling_func = method_map[method]

        if method == 'stratified':
            local_indices = sampling_func(clean_features, n_select=n_select, n_clusters=max(20, n_select // 15))
        else:
            local_indices = sampling_func(clean_features, n_select=n_select)

        selected_files = [clean_files[i] for i in local_indices]
        print(f"  采样完成: {len(selected_files)} 张")
        print(f"  耗时: {time.time()-t2:.1f}s")

    # Step 4: 质量评估
    print(f"\n[Step 4] 采样质量评估")
    if len(clean_features) > n_select:
        selected_global = np.array([i for i, f in enumerate(image_files) if f in set(selected_files)])
        evaluator = SamplingEvaluator(features, selected_global)
        evaluator.full_report()

        # 对比随机采样
        random_indices = np.random.choice(len(features), min(n_select, len(features)), replace=False)
        print(f"\n  [对比] 随机采样 {n_select} 张:")
        eval_random = SamplingEvaluator(features, random_indices)
        eval_random.full_report()
    else:
        print(f"  样本数不足，跳过评估")

    # Step 5: 可视化 (可选)
    if not no_vis:
        print(f"\n[Step 5] 可视化")
        try:
            from sklearn.decomposition import PCA
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            pca = PCA(n_components=2)
            coords = pca.fit_transform(features)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=10, alpha=0.3, label='全部样本')

            if len(clean_features) > n_select:
                selected_global = np.array([i for i, f in enumerate(image_files) if f in set(selected_files)])
                ax.scatter(coords[selected_global, 0], coords[selected_global, 1],
                          c='red', s=30, alpha=0.8, label=f'选中 ({len(selected_files)})')

            ax.legend()
            ax.set_title(f'Sample Selection ({method}, n={n_select})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')

            vis_path = os.path.join(dst_dir, 'selection_visualization.png')
            os.makedirs(dst_dir, exist_ok=True)
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  可视化已保存: {vis_path}")
        except Exception as e:
            print(f"  可视化跳过: {e}")

    # Step 6: 复制文件
    print(f"\n[Step 6] 复制选中样本到 {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)

    for src_path in selected_files:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)

    print(f"\n{'='*70}")
    print(f"完成! 从 {len(image_files)} 张 → {len(selected_files)} 张")
    print(f"输出目录: {dst_dir}")
    print(f"总耗时: {time.time()-t0:.1f}s")
    print(f"{'='*70}")

    return selected_files


def main():
    parser = argparse.ArgumentParser(
        description='从大量图像中筛选最具代表性的样本 (使用DINOv3特征+Hybrid采样)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python data_clening/select_samples.py --src_dir "C:\\Users\\finn.zhang\\project\\Datasets\\JieMei\\train\\good2_merge" --n_select 300

  # 指定输出目录和方法
  python data_clening/select_samples.py --src_dir ./data/good --n_select 200 --method coreset --dst_dir ./data/selected

  # GPU加速
  python data_clening/select_samples.py --src_dir ./data/good --n_select 300 --device cuda

  # 服务器环境（无GUI）
  python data_clening/select_samples.py --src_dir ./data/good --n_select 300 --no_vis
        """)

    parser.add_argument('--src_dir', type=str, required=True,
                        help='源图像目录')
    parser.add_argument('--dst_dir', type=str, default=None,
                        help='输出目录 (默认: src_dir_selected)')
    parser.add_argument('--n_select', type=int, default=300,
                        help='选择的样本数量 (default: 300)')
    parser.add_argument('--method', type=str, default='hybrid',
                        choices=['kmeans', 'coreset', 'stratified', 'hybrid'],
                        help='采样方法 (default: hybrid)')
    parser.add_argument('--model_name', type=str, default='dinov3_vitl16',
                        choices=['dinov3_vitb16', 'dinov3_vits16', 'dinov3_vitl16'],
                        help='特征提取模型 (default: dinov3_vitl16)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备: cpu 或 cuda (default: cuda)')
    parser.add_argument('--contamination', type=float, default=0,
                        help='异常检测清洗比例 (default: 0, 跳过清洗)')
    parser.add_argument('--no_vis', action='store_true', default=False,
                        help='不生成可视化图')

    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = args.src_dir.rstrip('/\\') + '_selected'

    select_representative_samples(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        n_select=args.n_select,
        method=args.method,
        model_name=args.model_name,
        device=args.device,
        contamination=args.contamination,
        no_vis=args.no_vis,
    )


if __name__ == '__main__':
    main()

    # python data_clening/select_samples.py --src_dir "C:\Users\finn.zhang\project\Datasets\hainanjiahua_20260428\OK_crop" --dst_dir "C:\Users\finn.zhang\project\Datasets\hainanjiahua_20260428\OK_crop_select_15" --n_select 15 --method hybrid --model_name dinov3_vitl16 --device cuda --contamination 0 --no_vis