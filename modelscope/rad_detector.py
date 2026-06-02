from typing import List, Optional, Dict, Union
import torch
import torch.nn.functional as F
from .dataset_v2 import get_gaussian_kernel


class RADDetector:
    def __init__(
        self,
        memory_bank: Dict[str, List[torch.Tensor]],
        layer_idx_list: List[int],
        k_image: int = 150,
        use_positional_bank: bool = True,
        pos_radius: int = 1,
        device: str = 'cuda',
        layer_weights: Optional[List[float]] = None,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
    ):
        self.memory_bank = memory_bank
        self.layer_idx_list = sorted(layer_idx_list)
        self.num_layers = len(self.layer_idx_list)
        self.k_image = k_image
        self.use_positional_bank = use_positional_bank
        self.pos_radius = pos_radius
        self.device = device

        self.cls_banks = [cb.to(device) for cb in memory_bank['cls']]
        self.patch_banks = [pb.to(device) for pb in memory_bank['patch']]

        self.cls_bank_global = self.cls_banks[-1]
        self.cls_bank_global_norm = F.normalize(self.cls_bank_global, dim=-1)

        if layer_weights is None or len(layer_weights) != self.num_layers:
            self.layer_weights = [1.0 / self.num_layers] * self.num_layers
        else:
            s = sum(layer_weights)
            if s <= 0:
                self.layer_weights = [1.0 / self.num_layers] * self.num_layers
            else:
                self.layer_weights = [w / s for w in layer_weights]

        self.gaussian_kernel = get_gaussian_kernel(
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma,
        ).to(device)

    def detect(
        self,
        patch_features_list: List[Union[torch.Tensor, tuple]],
        cls_features_list: Optional[List[torch.Tensor]] = None,
        resize_mask: int = 256,
    ):
        patch_list: List[torch.Tensor] = []
        cls_list: List[torch.Tensor] = []

        for feat in patch_features_list:
            if isinstance(feat, tuple):
                patch_list.append(feat[0])
                cls_list.append(feat[1])
            else:
                patch_list.append(feat)

        # Favor explicit cls_features_list; fall back to tuples parsed from patch_features_list
        if cls_features_list is not None:
            cls_list = cls_features_list

        first_patch = patch_list[0]
        B = first_patch.shape[0]
        L = first_patch.shape[1]
        h = w = int(L ** 0.5)

        if len(cls_list) == 0:
            raise ValueError("CLS features are required for image-level retrieval")

        cls_x_global = cls_list[-1]
        cls_x_global_norm = F.normalize(cls_x_global, dim=-1)
        sim_img = torch.matmul(cls_x_global_norm, self.cls_bank_global_norm.t())
        _, topk_idx = torch.topk(sim_img, k=min(self.k_image, sim_img.shape[1]), dim=-1)

        patch_scores_batch = []

        for b in range(B):
            neigh_indices = topk_idx[b]
            scores_per_layer = []

            for li in range(self.num_layers):
                patches_x_l = patch_list[li]
                bank_l = self.patch_banks[li]

                q_feat = patches_x_l[b]
                neigh_feat = bank_l[neigh_indices]

                q_feat = F.normalize(q_feat, dim=-1)
                neigh_feat = F.normalize(neigh_feat, dim=-1)

                if self.use_positional_bank:
                    patch_score_l = self._positional_knn(q_feat, neigh_feat, h, w)
                else:
                    bank_local = neigh_feat.reshape(-1, neigh_feat.shape[-1])
                    sim = torch.matmul(q_feat, bank_local.t())
                    nn_sim, _ = sim.max(dim=-1)
                    patch_score_l = 1.0 - nn_sim

                scores_per_layer.append(patch_score_l)

            fused = torch.zeros_like(scores_per_layer[0])
            for li in range(self.num_layers):
                fused = fused + self.layer_weights[li] * scores_per_layer[li]

            patch_scores_batch.append(fused)

        patch_scores_batch = torch.stack(patch_scores_batch, dim=0)
        patch_maps = patch_scores_batch.view(B, 1, h, w)

        anomaly_map = F.interpolate(
            patch_maps, size=resize_mask, mode='bilinear', align_corners=False
        )
        anomaly_map = self.gaussian_kernel(anomaly_map)

        if B == 1:
            anomaly_map_np = anomaly_map[0, 0].detach().cpu().numpy()
        else:
            anomaly_map_np = anomaly_map.mean(dim=0)[0].detach().cpu().numpy()

        return anomaly_map_np

    def _positional_knn(self, q_feat, neigh_feat, h, w):
        L = q_feat.shape[0]
        scores = torch.empty(L, device=self.device)

        for j in range(L):
            r, c = divmod(j, w)
            r_min, r_max = max(0, r - self.pos_radius), min(h - 1, r + self.pos_radius)
            c_min, c_max = max(0, c - self.pos_radius), min(w - 1, c + self.pos_radius)

            idx_list = [rr * w + cc for rr in range(r_min, r_max + 1) for cc in range(c_min, c_max + 1)]
            idx = torch.tensor(idx_list, device=self.device, dtype=torch.long)
            neigh_local = neigh_feat[:, idx, :].reshape(-1, neigh_feat.shape[-1])

            if neigh_local.numel() == 0:
                scores[j] = 1.0
                continue

            sim = torch.matmul(q_feat[j:j + 1], neigh_local.t())
            nn_sim = sim.max()
            scores[j] = 1.0 - nn_sim

        return scores
