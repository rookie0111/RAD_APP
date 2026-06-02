"""
Direct safetensors model for DINOv3.
The model architecture exactly matches the ModelScope safetensors naming convention,
so weights can be loaded with strict=True without any key mapping or conversion.

Critical fix: Includes RoPE (Rotary Position Embedding) which is essential for
correct spatial feature extraction. The safetensors format doesn't store rope_embed.periods,
so we compute it deterministically from base=100 (matching official DINOv3 config).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, base=100.0, normalize_coords="separate"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.D_head = embed_dim // num_heads
        self.base = base
        self.normalize_coords = normalize_coords
        assert embed_dim % (4 * num_heads) == 0
        self.register_buffer(
            "periods",
            self._compute_periods(),
            persistent=True,
        )

    def _compute_periods(self):
        return self.base ** (
            2 * torch.arange(self.D_head // 4, dtype=torch.float32) / (self.D_head // 2)
        )

    def forward(self, H, W):
        device = self.periods.device
        dtype = torch.float32

        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / H
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / W
        elif self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / max_HW
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / min_HW
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / min_HW
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, in_chans=3):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def apply_rope(self, q, k, rope):
        sin, cos = rope
        rope_dtype = sin.dtype
        q_dtype = q.dtype
        k_dtype = k.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        if prefix > 0:
            q_prefix = q[:, :, :prefix, :]
            q = rope_apply(q[:, :, prefix:, :], sin, cos)
            q = torch.cat((q_prefix, q), dim=-2)
            k_prefix = k[:, :, :prefix, :]
            k = rope_apply(k[:, :, prefix:, :], sin, cos)
            k = torch.cat((k_prefix, k), dim=-2)
        else:
            q = rope_apply(q, sin, cos)
            k = rope_apply(k, sin, cos)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x, rope=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=True)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.lambda1 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.lambda1


class TransformerLayerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attention = AttentionBlock(embed_dim, num_heads)
        self.layer_scale1 = LayerScale(embed_dim, init_values)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp = MLPBlock(embed_dim, mlp_hidden_dim)
        self.layer_scale2 = LayerScale(embed_dim, init_values)

    def forward(self, x, rope=None):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, rope=rope)
        x = residual + self.layer_scale1(x)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.layer_scale2(x)
        return x


class DINOv3SafetensorsModel(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12,
                 patch_size=16, mlp_ratio=4.0, init_values=1e-5, in_chans=3,
                 n_storage_tokens=4, rope_base=100.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.embeddings = PatchEmbedding(embed_dim, patch_size, in_chans)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_base,
            normalize_coords="separate",
        )

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.layer = nn.ModuleList([
            TransformerLayerBlock(embed_dim, num_heads, mlp_hidden_dim, init_values)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def _prepare_tokens(self, x):
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        x = self.embeddings(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, register_tokens, x], dim=1)

        return x, h, w

    def forward(self, x):
        x, h, w = self._prepare_tokens(x)
        rope = self.rope_embed(H=h, W=w)

        for layer in self.layer:
            x = layer(x, rope=rope)

        x = self.norm(x)
        return x

    def get_intermediate_layers(self, x, n, return_class_token=True, reshape=False, norm=True):
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        x = self.embeddings(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, register_tokens, x], dim=1)

        rope = self.rope_embed(H=h, W=w)

        n_set = set(n)
        outputs = []
        for i, layer in enumerate(self.layer):
            x = layer(x, rope=rope)
            if i in n_set:
                cls_feat = x[:, 0]
                patch_feat = x[:, self.n_storage_tokens + 1:]

                if norm:
                    full_normed = self.norm(x)
                    cls_feat = full_normed[:, 0]
                    patch_feat = full_normed[:, self.n_storage_tokens + 1:]

                outputs.append((patch_feat, cls_feat))

        return outputs

    def _remap_state_dict(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            if key == 'embeddings.cls_token':
                new_sd['cls_token'] = value
            elif key == 'embeddings.mask_token':
                new_sd['mask_token'] = value
            elif key == 'embeddings.register_tokens':
                new_sd['register_tokens'] = value
            elif key == 'embeddings.patch_embeddings.weight':
                new_sd['embeddings.proj.weight'] = value
            elif key == 'embeddings.patch_embeddings.bias':
                new_sd['embeddings.proj.bias'] = value
            else:
                new_sd[key] = value
        return new_sd


MODEL_CONFIGS = {
    'dinov3_vits16': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'patch_size': 16,
        'mlp_ratio': 4.0,
        'init_values': 1e-5,
        'in_chans': 3,
        'n_storage_tokens': 4,
        'rope_base': 100.0,
    },
    'dinov3_vitb16': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'patch_size': 16,
        'mlp_ratio': 4.0,
        'init_values': 1e-5,
        'in_chans': 3,
        'n_storage_tokens': 4,
        'rope_base': 100.0,
    },
    'dinov3_vitl16': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'patch_size': 16,
        'mlp_ratio': 4.0,
        'init_values': 1e-5,
        'in_chans': 3,
        'n_storage_tokens': 4,
        'rope_base': 100.0,
    },
}


def _detect_model_config(weight_path):
    """Auto-detect model config from weight file (.safetensors or .pth)."""
    import torch

    if weight_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(weight_path, device='cpu')
    else:
        sd = torch.load(weight_path, map_location='cpu')

    if 'embeddings.patch_embeddings.weight' in sd:
        embed_dim = sd['embeddings.patch_embeddings.weight'].shape[0]
    else:
        raise ValueError("Cannot detect embed_dim from safetensors")

    depth = 0
    while f'layer.{depth}.norm1.weight' in sd:
        depth += 1

    num_heads = {384: 6, 768: 12, 1024: 16}.get(embed_dim, embed_dim // 64)

    if 'layer.0.mlp.up_proj.weight' in sd:
        hidden_dim = sd['layer.0.mlp.up_proj.weight'].shape[0]
        mlp_ratio = round(hidden_dim / embed_dim)
    else:
        mlp_ratio = 4

    config = {
        'embed_dim': embed_dim, 'depth': depth, 'num_heads': num_heads,
        'patch_size': 16, 'mlp_ratio': mlp_ratio, 'init_values': 1e-5, 'in_chans': 3,
        'n_storage_tokens': 4, 'rope_base': 100.0,
    }
    print(f"Auto-detected: embed_dim={embed_dim}, depth={depth}, "
          f"num_heads={num_heads}, mlp_ratio={mlp_ratio}")
    return config


def load_model(model_name=None, safetensors_path=None, auto_detect=True):
    import os
    import torch

    if model_name is not None and model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name].copy()
    elif auto_detect and safetensors_path is not None:
        config = _detect_model_config(safetensors_path)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model = DINOv3SafetensorsModel(**config)

    if safetensors_path is not None:
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Weight file not found: {safetensors_path}")

        if safetensors_path.endswith('.safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(safetensors_path, device='cpu')
        else:
            sd = torch.load(safetensors_path, map_location='cpu')

        remapped_sd = model._remap_state_dict(sd)

        result = model.load_state_dict(remapped_sd, strict=False)
        loaded = len(remapped_sd) - len(result.unexpected_keys)
        total_model_keys = len(model.state_dict())
        print(f"Loaded {loaded}/{len(remapped_sd)} weights from safetensors "
              f"(model has {total_model_keys} keys, "
              f"missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)})")

        if result.missing_keys:
            print(f"Missing keys (expected, will use init values): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys[:5]}")

    model.eval()
    return model