# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import List, Optional, Union
from urllib.parse import urlparse
from pathlib import Path

import torch

DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_arch(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
):
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    return model_arch


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{model_arch}"
    model_filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: Optional[float] = None,
    pos_embed_rope_max_period: Optional[float] = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: Optional[float] = None,
    pos_embed_rope_jitter_coords: Optional[float] = None,
    pos_embed_rope_rescale_coords: Optional[float] = None,
    pos_embed_rope_dtype: str = "fp32",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: Optional[float] = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from dinov3.models.vision_transformer import DinoVisionTransformer

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed_rope_base=pos_embed_rope_base,
        pos_embed_rope_min_period=pos_embed_rope_min_period,
        pos_embed_rope_max_period=pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=pos_embed_rope_dtype,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale_init,
        norm_layer=norm_layer,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_vit_model_url(
                patch_size=patch_size,
                compact_arch_name=compact_arch_name,
                version=version,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()
    return model


def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""

    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: List[int] = [3, 3, 27, 3],
    dims: List[int] = [128, 256, 512, 1024],
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    **kwargs,
):
    from dinov3.models.convnext import ConvNeXt

    model_kwargs = dict(
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_convnext_model_url(
                compact_arch_name=compact_arch_name,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash = matches[0]
        if hash == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "46503df0"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vithplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.4,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_convnext_tiny(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "21b726bb"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from dinov3.models.convnext import convnext_sizes

    size_dict = convnext_sizes["tiny"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_tiny",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_small(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "296db49d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from dinov3.models.convnext import convnext_sizes

    size_dict = convnext_sizes["small"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_small",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from dinov3.models.convnext import convnext_sizes

    size_dict = convnext_sizes["base"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_large(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "61fa432d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from dinov3.models.convnext import convnext_sizes

    size_dict = convnext_sizes["large"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_large",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


# 模型名称到函数的映射
MODELS = {
    'dinov3_vits16': dinov3_vits16,
    'dinov3_vits16plus': dinov3_vits16plus,
    'dinov3_vitb16': dinov3_vitb16,
    'dinov3_vitl16': dinov3_vitl16,
    'dinov3_vitl16plus': dinov3_vitl16plus,
    'dinov3_vith16plus': dinov3_vith16plus,
    'dinov3_vit7b16': dinov3_vit7b16,
}

def _load_state_dict(weight_path: str, map_location="cpu") -> dict:
    """
    加载权重文件，支持：
    - 本地 .pth / .pt 文件
    - 本地 .safetensors 文件
    - ModelScope model_id（格式: "modelscope://owner/repo:filename"）
    """
    if weight_path.startswith("modelscope://"):
        # 格式: modelscope://owner/repo:filename
        # 例如: modelscope://damo/dinov3_vitb16:dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        path_part = weight_path[len("modelscope://"):]
        if ":" in path_part:
            repo_id, filename = path_part.rsplit(":", 1)
        else:
            repo_id, filename = path_part, None

        try:
            from modelscope.hub.snapshot_download import snapshot_download
            local_dir = snapshot_download(repo_id)
        except ImportError:
            raise ImportError("请先安装 modelscope: pip install modelscope")

        weight_path = os.path.join(local_dir, filename) if filename else local_dir

    if weight_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("请先安装 safetensors: pip install safetensors")
        return load_file(weight_path, device=str(map_location))

    return torch.load(weight_path, map_location=map_location)

def _create_key_mapping(state_dict, model_state_dict):
    """
    创建 safetensors/state_dict 键名到模型键名的映射
    处理 ModelScope safetensors 与 DINOv3 官方权重的键名差异

    完整映射规则:
    - embeddings.cls_token -> cls_token
    - embeddings.mask_token -> mask_token (需要 squeeze)
    - embeddings.register_tokens -> storage_tokens
    - embeddings.patch_embeddings.* -> patch_embed.proj.*
    - layer.N.attention.{q,k,v}_proj.* -> 已由 _fuse_qkv 融合为 blocks.N.attn.qkv.*
    - layer.N.attention.o_proj.* -> blocks.N.attn.proj.*
    - layer.N.layer_scale1.lambda1 -> blocks.N.ls1.gamma
    - layer.N.layer_scale2.lambda1 -> blocks.N.ls2.gamma
    - layer.N.mlp.up_proj.* -> blocks.N.mlp.fc1.*
    - layer.N.mlp.down_proj.* -> blocks.N.mlp.fc2.*
    - layer.N.norm1.* -> blocks.N.norm1.*
    - layer.N.norm2.* -> blocks.N.norm2.*
    - norm.* -> norm.*
    """
    mapping = {}
    import re

    def convert_key(sd_key):
        key = sd_key

        # embeddings 映射
        emb_map = {
            'embeddings.cls_token': 'cls_token',
            'embeddings.mask_token': 'mask_token',
            'embeddings.register_tokens': 'storage_tokens',
            'embeddings.patch_embeddings.weight': 'patch_embed.proj.weight',
            'embeddings.patch_embeddings.bias': 'patch_embed.proj.bias',
        }
        if key in emb_map:
            return emb_map[key]

        # layer.N.* -> blocks.N.*
        key = re.sub(r'^layer\.(\d+)\.', r'blocks.\1.', key)

        # attention.* -> attn.*
        key = key.replace('attention.', 'attn.')

        # attn.o_proj -> attn.proj
        key = key.replace('attn.o_proj.', 'attn.proj.')

        # layer_scale1.lambda1 -> ls1.gamma
        key = key.replace('layer_scale1.lambda1', 'ls1.gamma')
        # layer_scale2.lambda1 -> ls2.gamma
        key = key.replace('layer_scale2.lambda1', 'ls2.gamma')

        # mlp.up_proj -> mlp.fc1, mlp.down_proj -> mlp.fc2
        key = key.replace('mlp.up_proj.', 'mlp.fc1.')
        key = key.replace('mlp.down_proj.', 'mlp.fc2.')

        return key

    # 直接匹配 (官方 pth 格式)
    for sd_key in state_dict.keys():
        if sd_key in model_state_dict:
            if state_dict[sd_key].shape == model_state_dict[sd_key].shape:
                mapping[sd_key] = sd_key

    # 转换匹配 (safetensors 格式)
    for sd_key in state_dict.keys():
        if sd_key in mapping:
            continue
        model_key = convert_key(sd_key)
        if model_key in model_state_dict:
            if state_dict[sd_key].shape == model_state_dict[model_key].shape:
                mapping[sd_key] = model_key
            elif sd_key == 'embeddings.mask_token' and model_key == 'mask_token':
                mapping[sd_key] = model_key

    return mapping


def _fuse_qkv(state_dict, model_state_dict):
    """
    将分离的 q/k/v 权重融合成 qkv 格式
    ModelScope safetensors: attention.q_proj, attention.k_proj, attention.v_proj (分离)
    官方 DINOv3: attn.qkv (融合)

    支持两种键名格式:
    - safetensors: layer.N.attention.{q,k,v}_proj.{weight,bias}
    - 已转换: blocks.N.attn.{q,k,v}_proj.{weight,bias}

    注意: k_proj.bias 在 safetensors 中可能不存在（官方模型 qkv.bias 全为 0），
    此时用零向量填充。
    """
    import re

    # 同时匹配 safetensors 和已转换的键名
    patterns = [
        re.compile(r'layer\.(\d+)\.attention\.(q_proj|k_proj|v_proj)\.(weight|bias)'),
        re.compile(r'blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj)\.(weight|bias)'),
    ]

    qkv_data = {}
    keys_to_remove = []

    for sd_key in list(state_dict.keys()):
        for pattern in patterns:
            match = pattern.match(sd_key)
            if match:
                layer_idx, proj_type, param_type = match.groups()
                key = (int(layer_idx), param_type)

                if key not in qkv_data:
                    qkv_data[key] = {}

                qkv_data[key][proj_type] = state_dict[sd_key].clone()
                keys_to_remove.append(sd_key)
                break

    # 融合并创建 qkv
    for (layer_idx, param_type), data in qkv_data.items():
        q = data.get('q')
        k = data.get('k')
        v = data.get('v')

        # 如果缺少 k_proj（bias 场景），用零向量填充
        if q is not None and v is not None:
            if k is None:
                k = torch.zeros_like(q)

            fused = torch.cat([q, k, v], dim=0)

            model_key = f'blocks.{layer_idx}.attn.qkv.{param_type}'
            if model_key in model_state_dict:
                if fused.shape == model_state_dict[model_key].shape:
                    state_dict[model_key] = fused
                    print(f"融合: {model_key} - {fused.shape}")
                else:
                    print(f"警告: {model_key} shape 不匹配: fused={fused.shape}, model={model_state_dict[model_key].shape}")

    # 删除分离的 q/k/v 键
    for key in set(keys_to_remove):
        if key in state_dict:
            del state_dict[key]

    return state_dict


def _adapt_state_dict(state_dict: dict, model) -> dict:
    """
    适配 checkpoint 与当前模型的 state_dict 差异：
    1. 官方 pth 格式直接加载
    2. ModelScope safetensors 格式：键名映射 + q/k/v 融合 + shape 调整
    """
    model_sd = model.state_dict()

    # 检测是否为 safetensors 格式 (需要融合 q/k/v)
    needs_fusion = any(
        'attention.q_proj.weight' in k or 'attn.q_proj.weight' in k
        for k in state_dict.keys()
    )

    if needs_fusion:
        print("检测到分离的 q/k/v 权重，融合中...")
        state_dict = _fuse_qkv(state_dict, model_sd)

    # 创建键名映射
    key_mapping = _create_key_mapping(state_dict, model_sd)

    # 统计映射情况
    total_keys = len(state_dict)
    mapped_keys = len(key_mapping)
    print(f"权重映射: {mapped_keys}/{total_keys} 个键成功匹配")

    # 构建 adapted state_dict
    adapted = dict()
    for k, v in model_sd.items():
        adapted[k] = v.clone()

    for sd_key, model_key in key_mapping.items():
        sd_val = state_dict[sd_key]
        model_val = adapted[model_key]

        if sd_val.shape == model_val.shape:
            adapted[model_key] = sd_val.clone()
        elif sd_key == 'embeddings.mask_token' and model_key == 'mask_token':
            # mask_token: [1, 1, C] -> [1, C]
            adapted[model_key] = sd_val.reshape(model_val.shape).clone()
            print(f"  mask_token 调整: {sd_val.shape} -> {model_val.shape}")
        else:
            try:
                adapted[model_key] = sd_val.reshape(model_val.shape).clone()
                print(f"  {sd_key}: {sd_val.shape} -> {model_val.shape}")
            except RuntimeError:
                print(f"  警告: {sd_key} -> {model_key} shape 不匹配且无法 reshape")

    return adapted

def load_dinov3_model(model_name, pretrained_weight_path=None):
    """加载 DINOv3 模型
    支持:
        - .pth / .pt 文件
        - .safetensors 文件
        - ModelScope 路径 (modelscope://owner/repo:filename)

    Args:
        model_name: 模型名称 (如 'dinov3_vitl16', 'dinov3_vitb16', 'dinov3_vits16')
        pretrained_weight_path: 权重文件路径
    """
    assert model_name in MODELS, f"model name must be one of {list(MODELS.keys())}"
    model = MODELS[model_name](pretrained=False)

    if pretrained_weight_path is not None:
        state_dict = _load_state_dict(pretrained_weight_path)
        state_dict = _adapt_state_dict(state_dict, model)

        # 检查并修复 NaN 值
        nan_keys = [k for k, v in state_dict.items() if torch.isnan(v).any()]
        if nan_keys:
            print(f"警告: 发现 {len(nan_keys)} 个权重包含 NaN，已替换为 0")
            for k in nan_keys:
                state_dict[k] = torch.zeros_like(state_dict[k])

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"缺失的权重键 ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"多余的权重键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        print("模型加载成功!")

    return model


def load_dinov3_model_layers(model_name, layers_to_extract_from, pretrained_weight_path=None):
    """加载 DINOv3 模型（指定层）
    
    Args:
        model_name: 模型名称
        layers_to_extract_from: 要提取特征的层列表
        pretrained_weight_path: 权重文件路径
    """
    assert model_name in MODELS, f"model name must be one of {list(MODELS.keys())}"
    model = MODELS[model_name](pretrained=False, layers_to_extract_from=layers_to_extract_from)
    
    if pretrained_weight_path is not None:
        state_dict = _load_state_dict(pretrained_weight_path)
        state_dict = _adapt_state_dict(state_dict, model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"缺失的权重键 ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"多余的权重键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        print("模型加载成功!")
    
    return model
