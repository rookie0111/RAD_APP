# RAD_APP — 基于检索的工业异常检测系统

RAD (Retrieval-based Anomaly Detection) 使用预训练 DINOv3 视觉编码器提取特征，通过"先全局后局部"的两级检索机制发现异常。无需训练，构建正常特征记忆库后即可检测。

## 核心特点

- **无需训练**：直接使用预训练 DINOv3 编码器，无需微调
- **即插即用**：构建正常特征记忆库后即可检测异常
- **自动权重格式检测**：支持官方 `.pth` 和 ModelScope `.safetensors` 两种权重格式，自动识别
- **多模型支持**：ViT-S/16、ViT-B/16、ViT-L/16
- **ONNX 导出**：支持导出为 ONNX 格式，便于部署
- **GUI 界面**：PySide6 图形界面，支持记忆库构建、评估和单样本检测
- **数据清洗**：hybrid/CoreSet/kmeans 采样工具筛选代表性样本

## 项目结构

```
RAD_APP/
├── modelscope/              # Python 包（可 import）
│   ├── __init__.py          # 包导出
│   ├── config.py            # 集中配置（路径、常量、默认参数）
│   ├── model.py             # DINOv3SafetensorsModel（ModelScope 格式权重）
│   ├── loader.py            # detect_weight_format() + load_model() 自动路由
│   ├── rad_detector.py       # RADDetector 核心异常检测逻辑
│   ├── dataset_v2.py        # MVTecDataset、FeatureExtractor、build_memory_bank
│   └── backbones_copy.py    # 官方 DINOv3 .pth 加载器（需 dinov3 包）
├── scripts/                 # 入口脚本（需将项目根目录加入 sys.path）
│   ├── rad_gui.py            # PySide6 GUI（主入口）
│   ├── export_onnx.py       # ONNX 导出工具
│   ├── onnx_inference_full.py  # ONNX 推理 CLI
│   └── select_samples.py    # 数据清洗/采样工具
├── _legacy/                 # 旧版文件（保留参考）
├── AGENTS.md                # 开发者指南
└── README.md
```

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+（建议 CUDA 11.0+）

### 安装依赖

```bash
pip install torch torchvision numpy scipy scikit-learn Pillow tqdm matplotlib PySide6 opencv-python safetensors onnxruntime
```

### 预训练权重

将 DINOv3 权重文件放在 `RAD/pretrained/` 目录：

| 模型 | 文件名 |
|------|--------|
| ViT-B/16 | `pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| ViT-S/16 | `pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| ViT-L/16 | `dinov3/facebook/dinov3-vitl16-pretrain-lvd1689m/model.safetensors` |

> **注意**：`dinov3_vitl16_pretrain_lvd1689m.pth` 虽然是 `.pth` 后缀，但内部是 ModelScope 格式。系统会自动检测，无需手动指定。

## 快速开始

### 1. GUI 模式（推荐）

```bash
python scripts/rad_gui.py
```

GUI 提供三个功能标签页：
- **特征记忆库构建**：选择数据集和编码器，一键构建记忆库
- **性能评估**：在测试集上评估 AUROC、AP、F1 等指标
- **单样本检测**：选择图像和记忆库，实时检测并可视化

### 2. 命令行 — 构建记忆库

```python
from modelscope.dataset_v2 import MVTecDataset, build_memory_bank, FeatureExtractor

encoder = FeatureExtractor(model_name='dinov3_vitb16', pretrained_path='path/to/weight.pth')
dataset = MVTecDataset('./data/mvtec', 'metal_nut', split='train')
layer_indices, cls_banks, patch_banks = build_memory_bank(encoder.model, dataset, device='cuda')

import torch
torch.save({'layers': layer_indices, 'cls_banks': cls_banks, 'patch_banks': patch_banks}, 'bank/memory_bank.pth')
```

### 3. 命令行 — ONNX 导出

```bash
# 导出编码器 + 检测器（推荐分离模式）
python scripts/export_onnx.py --mode separated --encoder_name dinov3_vitb16

# 导出完整 pipeline
python scripts/export_onnx.py --mode full --encoder_name dinov3_vitb16

# ViT-L16 自动调整层索引
python scripts/export_onnx.py --mode separated --encoder_name dinov3_vitl16
```

### 4. 命令行 — ONNX 推理

```bash
# 单张图片
python scripts/onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg

# 批量推理 (GPU)
python scripts/onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image_dir ./data/test/ng --device cuda

# 调整阈值
python scripts/onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg --threshold 0.12 --save_vis
```

### 5. 数据清洗

```bash
python scripts/select_samples.py --src_dir ./data/good --n_select 300 --method hybrid --model_name dinov3_vitl16 --device cuda
```

采样方法：`kmeans` / `coreset` / `stratified` / `hybrid`（默认）

## 权重格式

系统支持两种不兼容的 DINOv3 架构：

| | 官方 (.pth) | ModelScope (.safetensors / 部分 .pth) |
|---|---|---|
| Attention 键名 | `blocks.N.attn.qkv` (融合) | `layer.N.attention.q_proj` (分离 q/k/v) |
| Token 键名 | `cls_token`, `storage_tokens` | `cls_token`, `register_tokens` |
| Patch embed 键名 | `patch_embed.proj.*` | `embeddings.patch_embeddings.*` |
| 加载器 | `backbones_copy.py` → `DinoVisionTransformer` | `model.py` → `DINOv3SafetensorsModel` |

`loader.detect_weight_format(path)` 自动检测权重格式，`loader.load_model()` 自动路由到正确的架构。

## 提取层索引

不同模型深度的默认提取层不同，错误索引会导致特征质量差：

| 模型 | 层数 | 默认提取层 |
|------|------|-------------|
| `dinov3_vits16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitb16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitl16` | 24 | `[5, 11, 17, 23]` |

配置文件 `modelscope/config.py` 中的 `LAYER_CONFIGS` 字典集中管理这些默认值。

## 预处理差异

| 场景 | 预处理 |
|------|--------|
| GUI 推理 | Resize → 512×512，不做 CenterCrop |
| 记忆库训练 | Resize → 512 → CenterCrop → 448 |
| ONNX 导出 | 默认 `img_size=448` |

**注意**：训练和推理的 crop 策略不一致会导致精度下降。

## 集中配置

所有分散的常量已集中到 `modelscope/config.py`：

- `IMAGENET_MEAN` / `IMAGENET_STD` — ImageNet 归一化参数
- `LAYER_CONFIGS` — 各模型的默认提取层
- `WEIGHT_SEARCH_PATHS` / `ENCODER_WEIGHT_MAP` — 权重文件搜索路径
- `find_weight_path()` — 自动查找模型权重
- `DEFAULT_*` — 所有默认参数（image_size, crop_size, k_image, threshold 等）

```python
from config import IMAGENET_MEAN, LAYER_CONFIGS, find_weight_path
# 或从包级别导入
from modelscope.config import IMAGENET_MEAN, LAYER_CONFIGS, find_weight_path
```

## numpy dtype 陷阱

`np.array([0.485, 0.456, 0.406])` 默认是 float64，但模型权重是 float32。构建归一化常量时必须显式指定 `dtype=np.float32`：

```python
# 错误
mean = np.array([0.485, 0.456, 0.406])  # float64!

# 正确（使用 config.py 提供的常量）
from config import IMAGENET_MEAN_NP  # 已确保 float32
```

## RoPE 位置编码

RoPE 位置编码对空间特征提取至关重要。safetensors 格式不存储 `rope_embed.periods`，`model.py` 从 `base=100` 确定性重算。请勿删除或跳过此计算。

## PyInstaller 打包

```bash
pyinstaller RAD_Detector.spec
```

关键配置：
- 必须排除 PyQt5：`excludes=['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui']`
- `datas` 必须包含 `modelscope`、`dinov3`、`pretrained` 目录

分发需要：`dist/RAD_Detector.exe` + `pretrained/` + `dinov3/` + `bank/`

## 数据集格式

项目支持 MVTec-AD 格式：

```
data_path/category/
  train/good/*.png           # 正常样本
  test/good/*.png            # 正常测试
  test/defect_type/*.png     # 异常测试
  ground_truth/defect_type/*_mask.png
```

## 性能参考

MVTec-AD metal_nut 类别（ViT-B/16）：

| 指标 | 图像级 | 像素级 |
|------|--------|--------|
| AUROC | 1.0000 | 0.9848 |
| AP | 1.0000 | 0.8881 |
| F1 | 1.0000 | 0.8896 |