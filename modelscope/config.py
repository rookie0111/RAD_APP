"""
Centralized configuration for RAD_APP.

All magic numbers, default paths, and model-specific constants live here.
Import from this module instead of duplicating values across files.
"""

import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_APP_DIR)

# Data directory: search in both RAD_APP/data/ and RAD_reproduction/
# This allows flexibility whether running from RAD_APP or alongside RAD_reproduction
_DATA_DIRS = [
    os.path.join(_APP_DIR, 'data'),
    os.path.join(_ROOT_DIR, 'RAD_reproduction'),
]


def _find_data_dir():
    """Return the first data directory that exists and contains pretrained weights."""
    for d in _DATA_DIRS:
        pretrained = os.path.join(d, 'pretrained')
        if os.path.isdir(pretrained):
            return d
    return _DATA_DIRS[-1]


DATA_DIR = _find_data_dir()

# ---------------------------------------------------------------------------
# ImageNet normalization (use dtype=np.float32 when constructing numpy arrays!)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_MEAN_NP = np.array(IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD_NP = np.array(IMAGENET_STD, dtype=np.float32)

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_NAMES = ['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16']

LAYER_CONFIGS = {
    'dinov3_vits16': [3, 6, 9, 11],
    'dinov3_vitb16': [3, 6, 9, 11],
    'dinov3_vitl16': [5, 11, 17, 23],
}

EMBED_DIMS = {
    'dinov3_vits16': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
}

# ---------------------------------------------------------------------------
# Weight search paths (checked in order; first existing file wins)
# Paths are relative to DATA_DIR (which resolves to RAD_reproduction/ or RAD_APP/data/)
# ---------------------------------------------------------------------------
WEIGHT_SEARCH_PATHS = {
    'dinov3_vitb16': [
        os.path.join(DATA_DIR, 'pretrained', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
    ],
    'dinov3_vitl16': [
        os.path.join(DATA_DIR, 'dinov3', 'facebook', 'dinov3-vitl16-pretrain-lvd1689m', 'model.safetensors'),
        os.path.join(DATA_DIR, 'pretrained', 'dinov3_vitl16_pretrain_lvd1689m.pth'),
    ],
    'dinov3_vits16': [
        os.path.join(DATA_DIR, 'pretrained', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
        os.path.join(DATA_DIR, 'dinov3', 'facebook', 'dinov3-vits16-pretrain-lvd1689m', 'model.safetensors'),
    ],
}

# Single best weight per model (for GUI weight-map display)
ENCODER_WEIGHT_MAP = {
    'dinov3_vitb16': os.path.join(DATA_DIR, 'pretrained', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),
    'dinov3_vits16': os.path.join(DATA_DIR, 'pretrained', 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
    'dinov3_vitl16': os.path.join(DATA_DIR, 'dinov3', 'facebook', 'dinov3-vitl16-pretrain-lvd1689m', 'model.safetensors'),
}


def find_weight_path(model_name, weight_path=None):
    """Resolve weight path for a model.

    If *weight_path* is given and exists, return it directly.
    Otherwise search WEIGHT_SEARCH_PATHS for the first existing file.
    """
    if weight_path is not None and os.path.exists(weight_path):
        return weight_path
    for p in WEIGHT_SEARCH_PATHS.get(model_name, []):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find weight file for {model_name}")


# ---------------------------------------------------------------------------
# Default inference / training parameters
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE = 512
DEFAULT_CROP_SIZE = 448
DEFAULT_K_IMAGE = 30
DEFAULT_POS_RADIUS = 1
DEFAULT_USE_POSITIONAL_BANK = True
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 1.0
DEFAULT_THRESHOLD = 0.10
DEFAULT_ONNX_IMG_SIZE = 448
DEFAULT_RESIZE_MASK = 256
DEFAULT_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Memory bank default path
# ---------------------------------------------------------------------------
DEFAULT_BANK_PATH = os.path.join(DATA_DIR, 'bank', 'memory_bank_official.pth')

# ---------------------------------------------------------------------------
# ONNX output directory
# ---------------------------------------------------------------------------
DEFAULT_ONNX_DIR = os.path.join(_APP_DIR, 'onnx')

# ---------------------------------------------------------------------------
# sys.path entries that scripts must add for dinov3 and modelscope imports
# ---------------------------------------------------------------------------
SYS_PATH_DIRS = [_APP_DIR, _ROOT_DIR, DATA_DIR]

# ---------------------------------------------------------------------------
# PyInstaller distribution directories
# ---------------------------------------------------------------------------
PYINSTALLER_EXCLUDES = ['PyQt5', 'PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui']
PYINSTALLER_DATAS = [
    ('modelscope', 'modelscope'),
    ('dinov3', 'dinov3'),
    ('pretrained', 'pretrained'),
]