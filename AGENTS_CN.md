# RAD_APP 开发者指南

## 项目结构

```
RAD_APP/
├── modelscope/          # Python 包（可通过 import modelscope 导入）
│   ├── __init__.py      # 汇总导出所有子模块 + 配置
│   ├── config.py        # 集中化常量和默认值
│   ├── model.py         # DINOv3SafetensorsModel（ModelScope 格式权重）
│   ├── loader.py        # detect_weight_format() + load_model() 自动路由
│   ├── rad_detector.py  # RADDetector — 核心异常检测逻辑
│   ├── dataset_v2.py    # MVTecDataset、FeatureExtractor、build_memory_bank
│   └── backbones.py     # 官方 DINOv3 .pth 加载器（需 dinov3 包）
├── scripts/             # 入口脚本（会将项目根目录 + 数据目录添加到 sys.path）
│   ├── rad_gui.py       # PySide6 GUI（面向用户的主入口）
│   ├── export_onnx.py   # ONNX 导出（支持 separated/full/dinov3/rad/all 模式）
│   ├── onnx_inference_full.py  # ONNX 推理 CLI
│   └── select_samples.py # 数据清洗：hybrid/CoreSet/kmeans 采样
├── _legacy/             # 已废弃文件，保留仅供参考
├── AGENTS.md
└── README.md
```

入口脚本会将 `RAD_APP/` 加入 `sys.path` 以便 `import modelscope`，将 `RAD/` 加入作为项目根目录，将 `RAD/RAD_reproduction/` 加入以便 `import dinov3`。

**同级依赖**：`backbones.py` 中 `from dinov3.models.vision_transformer` 的代码位于 `RAD_reproduction/dinov3/`。脚本必须将 `RAD_reproduction/` 加入 `sys.path`，否则官方格式的代码路径会失败。

**数据目录**：`config.py` 会自动探测预训练权重和记忆库文件的存放位置。它首先搜索 `RAD_APP/data/`，其次搜索 `RAD_reproduction/`。当前所有权重文件均位于 `RAD_reproduction/pretrained/` 和 `RAD_reproduction/dinov3/`。

## 集中化配置

所有之前分散的常量现在统一在 `modelscope/config.py` 中：

- `IMAGENET_MEAN` / `IMAGENET_STD` 及其 `_NP` 变体（float32 类型）
- `LAYER_CONFIGS` — 各模型默认的层索引
- `WEIGHT_SEARCH_PATHS` / `ENCODER_WEIGHT_MAP` — 权重文件路径解析
- `find_weight_path(model_name)` — 自动发现权重文件
- `DEFAULT_*` 参数（image_size、crop_size、k_image、threshold 等）

导入方式：`from modelscope.config import ...`（外部导入）或 `from .config import ...`（包内部导入）。

## 权重格式（关键）

项目中共存两种互不兼容的 DINOv3 架构：

| | 官方格式 (.pth) | ModelScope 格式 (.safetensors / 部分 .pth) |
|---|---|---|
| Attention 键名 | `blocks.N.attn.qkv`（融合） | `layer.N.attention.q_proj`（分离 q/k/v） |
| Token 键名 | `cls_token`, `storage_tokens` | `cls_token`, `register_tokens` |
| Patch embed 键名 | `patch_embed.proj.*` | `embeddings.patch_embeddings.*` |
| 加载器 | `backbones_copy.py` → `DinoVisionTransformer` | `model.py` → `DINOv3SafetensorsModel` |

`loader.detect_weight_format(path)` 通过检查权重键名来判断格式，返回 `'official'` 或 `'modelscope'`。`loader.load_model()` 自动将请求路由到正确的架构。

**陷阱**：`dinov3_vitl16_pretrain_lvd1689m.pth` 虽然是 `.pth` 后缀，但实际是 ModelScope 格式（415 个键 → 格式检测器可正确识别）。切勿仅凭文件扩展名判断格式。

## 层索引

层索引因模型深度而异。错误的索引 → 浅层特征 → 检测效果差。默认值位于 `config.py` 的 `LAYER_CONFIGS` 中：

| 模型 | 深度 | 默认层 |
|---|---|---|
| `dinov3_vits16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitb16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitl16` | 24 | `[5, 11, 17, 23]` |

## RoPE

RoPE 位置编码对正确的空间特征至关重要。safetensors 格式不存储 `rope_embed.periods`，因此 `model.py` 会根据 `base=100` 确定性地重新计算。请勿删除或跳过此计算。

## 预处理

- **GUI 推理**（`scripts/rad_gui.py`）：缩放至 512×512，不做 CenterCrop。归一化：ImageNet 均值/标准差。
- **记忆库训练**（`modelscope/dataset_v2.py` 中的 `MVTecDataset`）：先缩放到 `image_size`（默认 512），然后 CenterCrop 到 `crop_size`（默认 448）。训练与推理的 crop 策略不一致会导致精度下降。
- **ONNX 导出**（`scripts/export_onnx.py`）：默认 `img_size=448`。

## numpy 数据类型陷阱

`np.array([0.485, 0.456, 0.406])` 默认为 float64 类型，而模型权重是 float32。请使用 `config.py` 中的 `IMAGENET_MEAN_NP` / `IMAGENET_STD_NP`（已确认为 float32），或始终显式传入 `dtype=np.float32`。

## PyInstaller 打包

- 必须排除 PyQt5（`excludes=['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui']`），否则会与 PySide6 冲突。
- `datas` 必须包含 `modelscope`、`dinov3` 和 `pretrained` 目录。
- 构建命令：`pyinstaller RAD_Detector.spec`
- 分发文件：`dist/RAD_Detector.exe` + `pretrained/` + `dinov3/` + `bank/`

## 无测试 / 检查工具

项目未配置测试套件、代码检查器或类型检查器。验证修改请直接运行相应脚本（例如 `python scripts/export_onnx.py --mode separated`）。

## 关键入口文件

| 文件 | 用途 |
|---|---|
| `scripts/rad_gui.py` | PySide6 图形界面（面向用户的主入口） |
| `scripts/export_onnx.py` | ONNX 导出（`--mode full\|separated\|dinov3\|rad\|all`） |
| `scripts/onnx_inference_full.py` | ONNX 推理 CLI（单张图片或批量） |
| `scripts/select_samples.py` | 数据清洗：hybrid/CoreSet/kmeans 采样 |
| `modelscope/config.py` | 集中化常量和默认值 |
| `modelscope/model.py` | `DINOv3SafetensorsModel`（ModelScope 格式权重） |
| `modelscope/loader.py` | `detect_weight_format()` + `load_model()` 自动路由 |
| `modelscope/rad_detector.py` | `RADDetector` — 核心异常检测逻辑 |
| `modelscope/dataset_v2.py` | `MVTecDataset`、`FeatureExtractor`、`build_memory_bank`、`get_gaussian_kernel` |
| `modelscope/backbones_copy.py` | 官方 DINOv3 `.pth` 加载器（需 `dinov3` 包） |
