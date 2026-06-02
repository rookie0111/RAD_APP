"""
Loader for ModelScope DINOv3 models (safetensors and ModelScope-format pth files).

Provides a unified interface that auto-detects whether to use the official DINOv3
architecture (backbones.py) or the ModelScope-compatible architecture (modelscope/model.py)
based on the weight file's key format.

Detection logic:
- Official DINOv3 weights have keys like 'blocks.N.attn.qkv', 'rope_embed.periods'
- ModelScope weights have keys like 'layer.N.attention.q_proj', 'embeddings.register_tokens'

Both .pth and .safetensors files may contain either format!
"""

import os

from .config import WEIGHT_SEARCH_PATHS, find_weight_path


def detect_weight_format(weight_path):
    """Auto-detect whether weight file uses official DINOv3 or ModelScope format.

    Returns:
        'official' if keys match official DINOv3 (blocks.N.attn.qkv, rope_embed.periods, etc.)
        'modelscope' if keys match ModelScope format (layer.N.attention.q_proj, etc.)
    """
    if weight_path is None:
        return None

    if weight_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(weight_path, device='cpu')
    else:
        import torch
        sd = torch.load(weight_path, map_location='cpu')

    # Check for ModelScope-specific key patterns
    for key in sd:
        if 'layer.' in key and 'attention.' in key:
            return 'modelscope'
        if key.startswith('embeddings.register_tokens'):
            return 'modelscope'
        if key.startswith('embeddings.patch_embeddings'):
            return 'modelscope'

    # Check for official DINOv3 key patterns
    for key in sd:
        if key.startswith('blocks.') and 'attn.qkv' in key:
            return 'official'
        if key == 'rope_embed.periods':
            return 'official'

    # Heuristic: ModelScope files have ~415 keys for ViT-L, official have ~188
    if len(sd) > 200:
        return 'modelscope'

    return 'official'


def load_model(model_name=None, weight_path=None, device='cpu'):
    """Load a DINOv3 model, auto-detecting the correct architecture.

    This function automatically determines whether to use the official DINOv3
    architecture (backbones.py) or the ModelScope-compatible architecture
    (modelscope/model.py) based on the weight file format.

    Args:
        model_name: One of 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
        weight_path: Path to the weight file (.pth or .safetensors)
                     If None, uses default path for the model_name.
        device: Device to load the model to.

    Returns:
        model with get_intermediate_layers() method compatible with RAD detector.
    """
    if weight_path is None:
        weight_path = find_weight_path(model_name)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    fmt = detect_weight_format(weight_path)

    if fmt == 'modelscope':
        from .model import load_model as load_modelscope
        model = load_modelscope(model_name=model_name, safetensors_path=weight_path)
    else:
        from .backbones import load_dinov3_model
        model = load_dinov3_model(model_name, pretrained_weight_path=weight_path)

    return model


def is_safetensors_path(path):
    """Check if the path points to a safetensors file."""
    if path is None:
        return False
    return path.endswith('.safetensors')


def is_modelscope_format(weight_path):
    """Check if a weight file uses ModelScope format (separate q/k/v projections)."""
    return detect_weight_format(weight_path) == 'modelscope'


def load_dinov3_model_safetensors(model_name, safetensors_path):
    """Load a DINOv3 model from a ModelScope safetensors file.

    This creates a model architecture that exactly matches the safetensors
    naming convention (separate q/k/v projections, layer_scale1/lambda1, etc.)
    and loads weights with strict matching.

    Args:
        model_name: One of 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
        safetensors_path: Path to the safetensors file.

    Returns:
        model with get_intermediate_layers() method compatible with RAD detector.
    """
    from .model import load_model as load_modelscope

    if safetensors_path is None:
        safetensors_path = find_weight_path(model_name)

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    model = load_modelscope(model_name=model_name, safetensors_path=safetensors_path)
    return model