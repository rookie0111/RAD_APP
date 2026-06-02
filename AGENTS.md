# RAD_APP

## Project structure

```
RAD_APP/
├── modelscope/          # Python package (importable as `modelscope`)
│   ├── __init__.py      # Re-exports from all submodules + config
│   ├── config.py        # Centralized constants and defaults
│   ├── model.py         # DINOv3SafetensorsModel (ModelScope-format weights)
│   ├── loader.py         # detect_weight_format() + load_model() auto-routing
│   ├── rad_detector.py   # RADDetector — core anomaly detection logic
│   ├── dataset_v2.py    # MVTecDataset, FeatureExtractor, build_memory_bank
│   └── backbones.py     # Official DINOv3 .pth loader (requires dinov3 package)
├── scripts/             # Entry-point scripts (add project root + data dir to sys.path)
│   ├── rad_gui.py       # PySide6 GUI (main user-facing entry point)
│   ├── export_onnx.py   # ONNX export (separated/full/dinov3/rad/all)
│   ├── onnx_inference_full.py  # ONNX inference CLI
│   └── select_samples.py # Data cleaning: hybrid/CoreSet/kmeans sampling
├── _legacy/             # Deprecated files kept for reference
├── AGENTS.md
└── README.md
```

Scripts add `RAD_APP/` to `sys.path` for `import modelscope`, `RAD/` for project root, and `RAD/RAD_reproduction/` for `import dinov3`.

**Sibling dependency**: `backbones.py` imports `from dinov3.models.vision_transformer` which lives in `RAD_reproduction/dinov3/`. Scripts must add `RAD_reproduction/` to `sys.path` or the official-format code path will fail.

**Data directory**: `config.py` auto-detects where pretrained weights and bank files live. It searches `RAD_APP/data/` first, then `RAD_reproduction/`. All current weight files reside in `RAD_reproduction/pretrained/` and `RAD_reproduction/dinov3/`.

## Centralized config

All previously scattered constants are in `modelscope/config.py`:
- `IMAGENET_MEAN` / `IMAGENET_STD` + `_NP` variants (float32)
- `LAYER_CONFIGS` — per-model default layer indices
- `WEIGHT_SEARCH_PATHS` / `ENCODER_WEIGHT_MAP` — weight file resolution
- `find_weight_path(model_name)` — auto-discover weight files
- `DEFAULT_*` parameters (image_size, crop_size, k_image, threshold, etc.)

Import: `from modelscope.config import ...` (external) or `from .config import ...` (within package).

## Weight formats (critical)

Two incompatible DINOv3 architectures coexist:

| | Official (.pth) | ModelScope (.safetensors / some .pth) |
|---|---|---|
| Attention keys | `blocks.N.attn.qkv` (fused) | `layer.N.attention.q_proj` (separate q/k/v) |
| Token keys | `cls_token`, `storage_tokens` | `cls_token`, `register_tokens` |
| Patch embed keys | `patch_embed.proj.*` | `embeddings.patch_embeddings.*` |
| Loader | `backbones_copy.py` → `DinoVisionTransformer` | `model.py` → `DINOv3SafetensorsModel` |

`loader.detect_weight_format(path)` inspects weight keys and returns `'official'` or `'modelscope'`. `loader.load_model()` auto-routes to the correct architecture.

**Trap**: `dinov3_vitl16_pretrain_lvd1689m.pth` is ModelScope format despite the `.pth` extension (415 keys → detected correctly by format inspector). Never assume format from file extension alone.

## Layer indices

Layer indices differ by model depth. Wrong indices → shallow features → poor detection. Default values are in `config.py` `LAYER_CONFIGS`:

| Model | Depth | Default layers |
|---|---|---|
| `dinov3_vits16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitb16` | 12 | `[3, 6, 9, 11]` |
| `dinov3_vitl16` | 24 | `[5, 11, 17, 23]` |

## RoPE

RoPE positional embeddings are essential for correct spatial features. The safetensors format does not store `rope_embed.periods`, so `model.py` recomputes them deterministically from `base=100`. Do not remove or skip this computation.

## Preprocessing

- **GUI inference** (`scripts/rad_gui.py`): resize to 512×512, no CenterCrop. Normalization: ImageNet mean/std.
- **Memory bank training** (`modelscope/dataset_v2.py` `MVTecDataset`): resize to `image_size` (default 512), then CenterCrop to `crop_size` (default 448). Mismatch between training and inference crop strategy will degrade accuracy.
- **ONNX export** (`scripts/export_onnx.py`): default `img_size=448`.

## numpy dtype trap

`np.array([0.485, 0.456, 0.406])` is float64 by default. Model weights are float32. Use `IMAGENET_MEAN_NP` / `IMAGENET_STD_NP` from `config.py` (already float32), or always pass `dtype=np.float32` explicitly.

## PyInstaller packaging

- Must exclude PyQt5 (`excludes=['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui']`) or it conflicts with PySide6.
- `datas` must include `modelscope`, `dinov3`, and `pretrained` directories.
- Build: `pyinstaller RAD_Detector.spec`
- Distribution: `dist/RAD_Detector.exe` + `pretrained/` + `dinov3/` + `bank/`

## No test/lint/typecheck

No test suite, linter, or type checker is configured. Verify changes by running the relevant script directly (e.g. `python scripts/export_onnx.py --mode separated`).

## Key entry points

| File | Purpose |
|---|---|
| `scripts/rad_gui.py` | PySide6 GUI (main user-facing entry point) |
| `scripts/export_onnx.py` | ONNX export (`--mode full\|separated\|dinov3\|rad\|all`) |
| `scripts/onnx_inference_full.py` | ONNX inference CLI (single image or batch) |
| `scripts/select_samples.py` | Data cleaning: hybrid/CoreSet/kmeans sampling |
| `modelscope/config.py` | Centralized constants and defaults |
| `modelscope/model.py` | `DINOv3SafetensorsModel` (ModelScope-format weights) |
| `modelscope/loader.py` | `detect_weight_format()` + `load_model()` auto-routing |
| `modelscope/rad_detector.py` | `RADDetector` — core anomaly detection logic |
| `modelscope/dataset_v2.py` | `MVTecDataset`, `FeatureExtractor`, `build_memory_bank`, `get_gaussian_kernel` |
| `modelscope/backbones_copy.py` | Official DINOv3 `.pth` loader (requires `dinov3` package) |