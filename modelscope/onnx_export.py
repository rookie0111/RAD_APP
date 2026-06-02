"""
RAD + DINOv3 ONNX 导出库函数。

此模块包含所有 ONNX 导出的核心类和函数，供 scripts/export_onnx.py CLI 和 rad_gui.py GUI 使用。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple

from .config import (
    LAYER_CONFIGS, DEFAULT_ONNX_IMG_SIZE, DEFAULT_RESIZE_MASK,
    DEFAULT_BANK_PATH, DEFAULT_ONNX_DIR, find_weight_path,
)
from .dataset_v2 import get_gaussian_kernel


class RADONNXModel(nn.Module):
    """RAD 检测器的 ONNX 可导出版本"""

    def __init__(
        self,
        cls_banks,
        patch_banks,
        layer_weights,
        k_image=150,
        use_positional_bank=True,
        pos_radius=1,
        resize_mask=256,
        gaussian_kernel_size=5,
        gaussian_sigma=1.0,
    ):
        super().__init__()

        self.num_layers = len(cls_banks)
        self.k_image = k_image
        self.use_positional_bank = use_positional_bank
        self.pos_radius = pos_radius
        self.resize_mask = resize_mask

        cls_banks_tensor = torch.stack(cls_banks, dim=0)
        patch_banks_tensor = torch.stack(patch_banks, dim=0)

        self.register_buffer('cls_banks', cls_banks_tensor)
        self.register_buffer('patch_banks', patch_banks_tensor)
        self.register_buffer('layer_weights', torch.tensor(layer_weights, dtype=torch.float32))

        self.register_buffer('cls_bank_global', cls_banks_tensor[-1])
        self.register_buffer('cls_bank_global_norm', F.normalize(self.cls_bank_global, dim=-1))

        N = patch_banks[0].shape[0]
        L = patch_banks[0].shape[1]
        h = w = int(L ** 0.5)

        self.h = h
        self.w = w
        self.L = L
        self.N = N
        self.gaussian_kernel_size = gaussian_kernel_size

        if use_positional_bank:
            pos_indices, pos_counts = self._precompute_position_indices(h, w, pos_radius)
            self.register_buffer('pos_indices', pos_indices)
            self.register_buffer('pos_counts', pos_counts)

        gaussian = get_gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
        self.register_buffer('gaussian_kernel', gaussian.weight.data)

    def _precompute_position_indices(self, h, w, radius):
        L = h * w
        max_kpos = (2 * radius + 1) ** 2
        indices = torch.zeros(L, max_kpos, dtype=torch.long)
        counts = torch.zeros(L, dtype=torch.long)

        for j in range(L):
            r = j // w
            c = j % w

            r_min = max(0, r - radius)
            r_max = min(h - 1, r + radius)
            c_min = max(0, c - radius)
            c_max = min(w - 1, c + radius)

            idx_list = []
            for rr in range(r_min, r_max + 1):
                base = rr * w
                for cc in range(c_min, c_max + 1):
                    idx_list.append(base + cc)

            counts[j] = len(idx_list)
            for k, idx in enumerate(idx_list):
                indices[j, k] = idx

        return indices, counts

    def forward(self, patch_features, cls_features):
        num_layers, B, L, C = patch_features.shape

        cls_x_global = cls_features[-1]
        cls_x_global_norm = F.normalize(cls_x_global, dim=-1)
        sim_img = torch.matmul(cls_x_global_norm, self.cls_bank_global_norm.t())
        k = min(self.k_image, sim_img.shape[1])
        _, topk_idx = torch.topk(sim_img, k=k, dim=-1)

        results = []

        for b in range(B):
            neigh_indices = topk_idx[b]

            layer_scores = []
            for li in range(self.num_layers):
                q_feat = patch_features[li, b]
                bank_l = self.patch_banks[li]
                neigh_feat = bank_l[neigh_indices]

                q_feat = F.normalize(q_feat, dim=-1)
                neigh_feat = F.normalize(neigh_feat, dim=-1)

                if self.use_positional_bank:
                    patch_score_l = self._positional_knn_vectorized(q_feat, neigh_feat)
                else:
                    patch_score_l = self._global_knn(q_feat, neigh_feat)

                layer_scores.append(patch_score_l)

            fused = torch.zeros(L, device=patch_features.device, dtype=patch_features.dtype)
            for li in range(self.num_layers):
                fused = fused + self.layer_weights[li] * layer_scores[li]

            results.append(fused)

        patch_scores = torch.stack(results, dim=0)
        patch_maps = patch_scores.view(B, 1, self.h, self.w)

        anomaly_map = F.interpolate(
            patch_maps,
            size=self.resize_mask,
            mode='bilinear',
            align_corners=False
        )

        anomaly_map = F.conv2d(anomaly_map, self.gaussian_kernel, padding=self.gaussian_kernel_size // 2)

        return anomaly_map.squeeze(1)

    def _positional_knn_vectorized(self, q_feat, neigh_feat):
        K = neigh_feat.shape[0]
        L = q_feat.shape[0]
        C = q_feat.shape[1]

        sim_all = torch.matmul(q_feat, neigh_feat.reshape(-1, C).t())
        nn_idx = sim_all.argmax(dim=-1)

        row_idx = torch.arange(L, device=q_feat.device)
        nn_sim = sim_all[row_idx, nn_idx]

        scores = 1.0 - nn_sim

        if self.use_positional_bank and hasattr(self, 'pos_indices'):
            max_kpos = self.pos_indices.shape[1]
            src_local = torch.arange(max_kpos, device=q_feat.device).unsqueeze(0).expand(L, -1)
            src_valid = src_local < self.pos_counts.unsqueeze(1)

            gathered = neigh_feat[:, self.pos_indices, :]
            gathered = gathered.permute(1, 0, 2, 3).reshape(L, K, max_kpos, C)
            normed = F.normalize(gathered, dim=-1)

            q_expand = q_feat.unsqueeze(2)
            local_sim = (q_expand * normed).sum(dim=-1)
            local_sim = local_sim.masked_fill(~src_valid.unsqueeze(1), float('-inf'))
            local_nn_sim = local_sim.max(dim=-1).values
            pos_nn_sim = local_nn_sim.max(dim=0).values

            scores = 1.0 - pos_nn_sim

        return scores

    def _global_knn(self, q_feat, neigh_feat):
        C = q_feat.shape[1]
        bank_local = neigh_feat.reshape(-1, C)
        sim = torch.matmul(q_feat, bank_local.t())
        nn_sim, _ = sim.max(dim=-1)
        return 1.0 - nn_sim


class DINOv3ONNXSimple(nn.Module):
    """DINOv3 ONNX 包装 - 自动适配官方和 ModelScope 架构"""

    def __init__(self, encoder, layer_indices: List[int]):
        super().__init__()
        self.encoder = encoder
        self.layer_indices = sorted(layer_indices)
        self.num_layers = len(layer_indices)

        self.embed_dim = encoder.embed_dim
        self.n_storage_tokens = encoder.n_storage_tokens

        # Detect architecture type
        self._is_modelscope = hasattr(encoder, 'layer') and not hasattr(encoder, 'blocks')

        if self._is_modelscope:
            # ModelScope architecture
            self.blocks = list(encoder.layer)
            self.patch_embed_proj = encoder.embeddings.proj
            self.cls_token_param = encoder.cls_token
            self.register_tokens_param = encoder.register_tokens
            self.final_norm = encoder.norm
            self.rope_embed = encoder.rope_embed

            patch_size = encoder.patch_size
            self._patch_size = patch_size
        else:
            # Official architecture
            self.blocks = list(encoder.blocks)
            self.patch_embed = encoder.patch_embed
            self.cls_token_param = encoder.cls_token
            self.register_tokens_param = encoder.storage_tokens
            self.final_norm = encoder.norm
            self.cls_norm = encoder.cls_norm if hasattr(encoder, 'cls_norm') else None
            self.untie_cls_and_patch_norms = getattr(encoder, 'untie_cls_and_patch_norms', False)
            self.rope_embed = getattr(encoder, 'rope_embed', None)
            self._patch_size = encoder.patch_embed.patch_size[0] if hasattr(encoder.patch_embed, 'patch_size') else 16

        # Pre-compute RoPE for a fixed image size (will be overridden at runtime)
        self._cached_h = None
        self._cached_w = None
        self._cached_rope = None

    def _get_rope(self, h, w, device):
        if self._cached_h == h and self._cached_w == w:
            return self._cached_rope

        if self.rope_embed is not None:
            rope = self.rope_embed(H=h, W=w)
            self._cached_h = h
            self._cached_w = w
            self._cached_rope = rope
            return rope
        return None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        H = x.shape[2] // self._patch_size
        W = x.shape[3] // self._patch_size

        if self._is_modelscope:
            # ModelScope: embeddings.proj -> flatten -> cat tokens
            x = self.patch_embed_proj(x)
            x = x.flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token_param.expand(B, -1, -1)
            register_tokens = self.register_tokens_param.expand(B, -1, -1)
            x = torch.cat([cls_tokens, register_tokens, x], dim=1)
        else:
            # Official: patch_embed (handles flatten internally) -> cat tokens
            x = self.patch_embed(x)
            x = x.flatten(1, 2)
            cls_token = self.cls_token_param.expand(B, -1, -1)
            if self.n_storage_tokens > 0:
                storage_tokens = self.register_tokens_param.expand(B, -1, -1)
            else:
                storage_tokens = torch.empty(B, 0, self.embed_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([cls_token, storage_tokens, x], dim=1)

        rope = self._get_rope(H, W, x.device)

        layer_outputs = []
        n_prefix = 1 + self.n_storage_tokens

        for i, blk in enumerate(self.blocks):
            if self._is_modelscope:
                x = blk(x, rope=rope)
            else:
                x = blk(x, rope)

            if i in self.layer_indices:
                layer_outputs.append(x)

        patch_features_list = []
        cls_features_list = []

        for out in layer_outputs:
            if not self._is_modelscope and self.untie_cls_and_patch_norms:
                x_norm_cls_reg = self.cls_norm(out[:, :n_prefix])
                x_norm_patch = self.final_norm(out[:, n_prefix:])
                out_normed = torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
            else:
                out_normed = self.final_norm(out)

            cls_tok = out_normed[:, 0]
            patch_tok = out_normed[:, n_prefix:]

            cls_features_list.append(cls_tok)
            patch_features_list.append(patch_tok)

        patch_features = torch.stack(patch_features_list, dim=0)
        cls_features = torch.stack(cls_features_list, dim=0)

        return patch_features, cls_features


class FullPipelineONNX(nn.Module):
    """完整的 RAD Pipeline ONNX 模型"""

    def __init__(self, encoder_wrapper: nn.Module, rad_model: nn.Module):
        super().__init__()
        self.encoder = encoder_wrapper
        self.rad = rad_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_features, cls_features = self.encoder(x)
        anomaly_map = self.rad(patch_features, cls_features)
        return anomaly_map


def load_encoder(encoder_name, weight_path, device='cpu'):
    """自动检测权重格式并加载编码器"""
    from .loader import load_model as load_model_auto, detect_weight_format

    fmt = detect_weight_format(weight_path)
    print(f"权重格式: {fmt}")

    model = load_model_auto(model_name=encoder_name, weight_path=weight_path, device=device)
    model.eval()
    return model


def export_dinov3_onnx(encoder, output_path, layer_indices, img_size=448,
                       opset_version=14, verify=True, device='cpu'):
    """导出 DINOv3 编码器为 ONNX 格式"""
    print(f"\n{'='*70}")
    print("导出 DINOv3 编码器为 ONNX")
    print(f"{'='*70}")

    encoder.eval()

    wrapper = DINOv3ONNXSimple(encoder, layer_indices)
    wrapper.to(device)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    print(f"\n输入尺寸: [B, 3, {img_size}, {img_size}]")
    print(f"提取层: {layer_indices}")

    with torch.no_grad():
        patch_out, cls_out = wrapper(dummy_input)

    print(f"输出 patch_features: {patch_out.shape}")
    print(f"输出 cls_features: {cls_out.shape}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    print(f"\n导出 ONNX 模型...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['patch_features', 'cls_features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'patch_features': {1: 'batch_size'},
            'cls_features': {1: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"ONNX 模型已保存: {output_path}")

    if verify:
        print("\n验证输出一致性...")
        import onnxruntime as ort

        with torch.no_grad():
            torch_patch, torch_cls = wrapper(dummy_input)

        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_outputs = sess.run(None, {'image': dummy_input.numpy()})

        onnx_patch, onnx_cls = onnx_outputs

        patch_diff = np.abs(torch_patch.numpy() - onnx_patch).max()
        cls_diff = np.abs(torch_cls.numpy() - onnx_cls).max()

        print(f"  patch_features 最大差异: {patch_diff:.6e}")
        print(f"  cls_features 最大差异: {cls_diff:.6e}")

        if patch_diff < 1e-3 and cls_diff < 1e-3:
            print("  ✓ 验证通过！")
            return True
        else:
            print("  ✗ 验证失败！差异过大")
            return False

    return True


def export_rad_onnx(memory_bank_path, output_path, k_image=150, use_positional_bank=True,
                    pos_radius=1, resize_mask=256, layer_weights=None,
                    opset_version=14, verify=True, device='cpu'):
    """导出 RAD 检测器为 ONNX 格式"""
    print(f"\n{'='*70}")
    print("导出 RAD 检测器为 ONNX")
    print(f"{'='*70}")

    print(f"\n加载记忆库: {memory_bank_path}")
    bank_data = torch.load(memory_bank_path, map_location='cpu')

    layer_indices = bank_data['layers']
    cls_banks = bank_data['cls_banks']
    patch_banks = bank_data['patch_banks']

    num_layers = len(layer_indices)
    print(f"层数: {num_layers}, 层索引: {layer_indices}")

    for li, layer in enumerate(layer_indices):
        print(f"  层 {layer}: CLS {cls_banks[li].shape}, Patch {patch_banks[li].shape}")

    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        s = sum(layer_weights)
        if s <= 0:
            layer_weights = [1.0 / num_layers] * num_layers
        else:
            layer_weights = [w / s for w in layer_weights]

    print(f"层权重: {layer_weights}")

    model = RADONNXModel(
        cls_banks=cls_banks,
        patch_banks=patch_banks,
        layer_weights=layer_weights,
        k_image=k_image,
        use_positional_bank=use_positional_bank,
        pos_radius=pos_radius,
        resize_mask=resize_mask
    )
    model.to(device)
    model.eval()

    N = patch_banks[0].shape[0]
    L = patch_banks[0].shape[1]
    C = patch_banks[0].shape[2]
    B = 1

    dummy_patch = torch.randn(num_layers, B, L, C, device=device)
    dummy_cls = torch.randn(num_layers, B, C, device=device)

    print(f"\n输入 patch_features: [{num_layers}, B, {L}, {C}]")
    print(f"输入 cls_features: [{num_layers}, B, {C}]")
    print(f"输出 anomaly_map: [B, {resize_mask}, {resize_mask}]")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    print(f"\n导出 ONNX 模型...")
    torch.onnx.export(
        model,
        (dummy_patch, dummy_cls),
        output_path,
        input_names=['patch_features', 'cls_features'],
        output_names=['anomaly_map'],
        dynamic_axes={
            'patch_features': {1: 'batch_size'},
            'cls_features': {1: 'batch_size'},
            'anomaly_map': {0: 'batch_size'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"ONNX 模型已保存: {output_path}")

    if verify:
        print("\n验证输出一致性...")
        import onnxruntime as ort

        with torch.no_grad():
            torch_output = model(dummy_patch, dummy_cls)

        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_output = sess.run(
            None,
            {
                'patch_features': dummy_patch.numpy(),
                'cls_features': dummy_cls.numpy()
            }
        )[0]

        torch_output_np = torch_output.numpy()

        max_diff = np.abs(torch_output_np - onnx_output).max()
        mean_diff = np.abs(torch_output_np - onnx_output).mean()

        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")

        if max_diff < 1e-4:
            print("  ✓ 验证通过！")
            return True
        else:
            print("  ✗ 验证失败！")
            return False

    return True


def export_full_pipeline_onnx(encoder, memory_bank_path, output_path, layer_indices,
                               img_size=448, k_image=150, use_positional_bank=True,
                               pos_radius=1, resize_mask=256, layer_weights=None,
                               opset_version=14, verify=True, device='cpu', low_memory=False):
    """导出完整的 RAD Pipeline 为单个 ONNX 模型"""
    print(f"\n{'='*70}")
    print("导出完整 RAD Pipeline 为 ONNX")
    print(f"{'='*70}")

    print("\n注意: 完整 Pipeline 模型较大，需要较多内存。")
    print("如果内存不足，建议使用分离模式（separated）代替。")

    encoder.eval()

    print(f"\n加载记忆库: {memory_bank_path}")
    bank_data = torch.load(memory_bank_path, map_location='cpu')

    bank_layer_indices = bank_data['layers']
    cls_banks = bank_data['cls_banks']
    patch_banks = bank_data['patch_banks']

    num_layers = len(bank_layer_indices)
    print(f"层数: {num_layers}, 层索引: {bank_layer_indices}")

    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        s = sum(layer_weights)
        if s <= 0:
            layer_weights = [1.0 / num_layers] * num_layers
        else:
            layer_weights = [w / s for w in layer_weights]

    encoder_wrapper = DINOv3ONNXSimple(encoder, layer_indices)

    rad_model = RADONNXModel(
        cls_banks=cls_banks,
        patch_banks=patch_banks,
        layer_weights=layer_weights,
        k_image=k_image,
        use_positional_bank=use_positional_bank,
        pos_radius=pos_radius,
        resize_mask=resize_mask
    )

    full_model = FullPipelineONNX(encoder_wrapper, rad_model)
    full_model.to(device)
    full_model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    print(f"\n输入尺寸: [B, 3, {img_size}, {img_size}]")
    print(f"输出尺寸: [B, {resize_mask}, {resize_mask}]")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    print(f"\n导出 ONNX 模型（这可能需要几分钟）...")

    try:
        torch.onnx.export(
            full_model,
            dummy_input,
            output_path,
            input_names=['image'],
            output_names=['anomaly_map'],
            dynamic_axes=None, # tensorrt 不支持动态batch
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )

        print(f"ONNX 模型已保存: {output_path}")

    except MemoryError:
        print("\n错误: 内存不足！")
        print("建议解决方案:")
        print("  1. 使用分离模式代替: python export_onnx.py --mode separated")
        print("  2. 增加系统可用内存")
        print("  3. 使用更小的模型（如 dinov3_vits16）")
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n错误: 内存不足！")
            print("建议解决方案:")
            print("  1. 使用分离模式代替: python export_onnx.py --mode separated")
            print("  2. 增加系统可用内存")
            print("  3. 使用更小的模型（如 dinov3_vits16）")
            return False
        raise

    if verify and not low_memory:
        print("\n验证输出一致性...")
        import onnxruntime as ort

        with torch.no_grad():
            torch_output = full_model(dummy_input)

        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_output = sess.run(None, {'image': dummy_input.numpy()})[0]

        max_diff = np.abs(torch_output.numpy() - onnx_output).max()
        mean_diff = np.abs(torch_output.numpy() - onnx_output).mean()

        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")

        if max_diff < 1e-3:
            print("  ✓ 验证通过！")
            return True
        else:
            print("  ✗ 验证失败！差异过大")
            return False

    return True
