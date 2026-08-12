r"""
图像匹配工具：给定一张图，在指定文件夹中找出最相似的 N 张图

使用 DINOv3 多尺度融合 CLS 特征 + 余弦相似度进行匹配。

用法:
  python data_clening/image_match.py --query path/to/query.jpg --target_dir path/to/folder

  # 指定返回数量和模型
  python data_clening/image_match.py --query ./data/ref.png --target_dir ./data/gallery --top_k 10 --model_name dinov3_vitb16

依赖:
  安装项目根目录的 requirements.txt，确保 pretrained/ 或 dinov3/ 下有权重文件。
"""

import os
import sys
import argparse
import numpy as np

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEIGHT_SEARCH_PATHS = {
    'dinov3_vitb16': [
        os.path.join(_parent_dir, 'pretrained', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
    ],
    'dinov3_vitl16': [
        os.path.join(_parent_dir, 'dinov3', 'facebook', 'dinov3-vitl16-pretrain-lvd1689m', 'model.safetensors'),
        os.path.join(_parent_dir, 'pretrained', 'dinov3_vitl16_pretrain_lvd1689m.pth'),
    ],
    'dinov3_vits16': [
        os.path.join(_parent_dir, 'pretrained', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
        os.path.join(_parent_dir, 'dinov3', 'facebook', 'dinov3-vits16-pretrain-lvd1689m', 'model.safetensors'),
    ],
}

LAYER_CONFIGS = {
    'dinov3_vitb16': [3, 5, 7, 9],
    'dinov3_vits16': [3, 6, 9, 11],
    'dinov3_vitl16': [5, 11, 17, 23],
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def load_model(model_name='dinov3_vitl16', device='cuda'):
    """加载 DINOv3 模型并返回 (model, format)"""
    import torch
    from modelscope.loader import load_model as _load_model, detect_weight_format

    weight_path = None
    for p in WEIGHT_SEARCH_PATHS.get(model_name, []):
        if os.path.exists(p):
            weight_path = p
            break
    if weight_path is None:
        raise FileNotFoundError(f"未找到 {model_name} 的权重文件，搜索路径: {WEIGHT_SEARCH_PATHS.get(model_name, [])}")

    fmt = detect_weight_format(weight_path)
    print(f"加载模型: {model_name} (格式: {fmt}, 设备: {device})")
    model = _load_model(model_name=model_name, weight_path=weight_path)
    model.to(device).eval()
    return model, fmt


def _get_layer_n(model, fmt, model_name):
    """根据模型格式返回 get_intermediate_layers 的 n 参数。
    - official 格式: n=4 (int, 最后4层)
    - modelscope 格式: n=[5,11,17,23] 等 (list, 指定层索引)
    """
    layer_indices = LAYER_CONFIGS.get(model_name, [5, 11, 17, 23])
    if fmt == 'modelscope':
        return layer_indices
    else:
        return len(layer_indices)


def extract_single_feature(model, image_path, fmt, device='cuda', img_size=512):
    """提取单张图像的融合 CLS 特征（L2 归一化后返回）"""
    import torch
    from PIL import Image

    model_name = getattr(model, 'model_name', 'dinov3_vitl16')
    n = _get_layer_n(model, fmt, model_name)

    img = Image.open(image_path).convert('RGB').resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean_np = np.array(IMAGENET_MEAN, dtype=np.float32)
    std_np = np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(((arr - mean_np) / std_np).transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        inter = model.get_intermediate_layers(tensor, n=n, return_class_token=True)
        layer_feats = []
        for _patch_feat, cls_feat in inter:
            layer_feats.append(cls_feat)
        fused = torch.cat(layer_feats, dim=-1)
        feat = fused[0].cpu().numpy()

    # L2 归一化
    norm = np.linalg.norm(feat)
    if norm > 1e-8:
        feat = feat / norm
    return feat


def extract_batch_features(model, image_paths, fmt, device='cuda', img_size=512, batch_size=8):
    """批量提取融合 CLS 特征（L2 归一化后返回）"""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    model_name = getattr(model, 'model_name', 'dinov3_vitl16')
    n = _get_layer_n(model, fmt, model_name)

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
            mean_np = np.array(IMAGENET_MEAN, dtype=np.float32)
            std_np = np.array(IMAGENET_STD, dtype=np.float32)
            tensor = torch.from_numpy(((arr - mean_np) / std_np).transpose(2, 0, 1).astype(np.float32))
            return tensor

    dataset = ImageList(image_paths, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inter = model.get_intermediate_layers(batch, n=n, return_class_token=True)
            layer_feats = []
            for _patch_feat, cls_feat in inter:
                layer_feats.append(cls_feat)
            fused = torch.cat(layer_feats, dim=-1)
            all_features.append(fused.cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    # L2 归一化
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features = features / norms
    return features


def find_similar_images(query_path, target_dir, top_k=5, model_name='dinov3_vitl16',
                        device='cuda', img_size=512):
    """在目标文件夹中找出与查询图最相似的 top_k 张图。
    
    Args:
        query_path: 查询图像路径
        target_dir: 目标图像文件夹
        top_k: 返回的最相似图像数量
        model_name: DINOv3 模型名
        device: 计算设备
        img_size: 输入图像尺寸
    
    Returns:
        list[dict]: [{'path': str, 'similarity': float}, ...] 按相似度从高到低排序
    """
    # 加载模型（只加载一次）
    model, fmt = load_model(model_name, device)

    # 提取查询图特征
    print(f"提取查询图特征: {query_path}")
    query_feat = extract_single_feature(model, query_path, fmt, device, img_size)

    # 扫描目标文件夹
    target_paths = sorted([
        os.path.join(target_dir, f) for f in os.listdir(target_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ])
    if len(target_paths) == 0:
        print("目标文件夹中没有图像文件")
        return []

    print(f"提取目标文件夹特征: {len(target_paths)} 张图像")
    target_feats = extract_batch_features(model, target_paths, fmt, device, img_size)

    # 计算余弦相似度（特征已 L2 归一化，点积即余弦相似度）
    sims = np.dot(target_feats, query_feat)

    # 排序取 top_k
    top_indices = np.argsort(-sims)[:min(top_k, len(sims))]
    results = []
    for idx in top_indices:
        results.append({
            'path': target_paths[idx],
            'similarity': float(sims[idx]),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='图像匹配：找出文件夹中最相似的图像')
    parser.add_argument('--query', required=True, help='查询图像路径')
    parser.add_argument('--target_dir', required=True, help='目标图像文件夹路径')
    parser.add_argument('--top_k', type=int, default=5, help='返回的最相似图像数量（默认5）')
    parser.add_argument('--model_name', default='dinov3_vitb16',
                        choices=['dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vits16'],
                        help='DINOv3 模型名（默认 vitl16）')
    parser.add_argument('--device', default='cuda', help='计算设备（cuda / cpu）')
    parser.add_argument('--img_size', type=int, default=448, help='输入图像尺寸（默认 512）')
    args = parser.parse_args()

    if not os.path.exists(args.query):
        print(f"错误: 查询图像不存在: {args.query}")
        sys.exit(1)

    if not os.path.isdir(args.target_dir):
        print(f"错误: 目标文件夹不存在: {args.target_dir}")
        sys.exit(1)

    results = find_similar_images(
        query_path=args.query,
        target_dir=args.target_dir,
        top_k=args.top_k,
        model_name=args.model_name,
        device=args.device,
        img_size=args.img_size,
    )

    print(f"\n{'='*60}")
    print(f"查询图: {args.query}")
    print(f"目标文件夹: {args.target_dir} ({len(results) if results else 0} 个结果)")
    print(f"{'='*60}")

    for i, r in enumerate(results):
        print(f"[{i+1}] 相似度: {r['similarity']*100:.1f}%  →  {r['path']}")

    if len(results) == 0:
        print("未找到任何匹配结果。")


if __name__ == '__main__':
    main()
