import torch
import torch.nn.functional as F
from .dataset_v2 import get_gaussian_kernel


class RADDetector:
    """
    RAD异常检测器
    
    核心流程：
    1. 使用最高层 CLS token 进行图像级检索
    2. 对每层的 patch tokens 进行最近邻搜索
    3. 多层融合得到最终异常图
    4. Gaussian平滑
    5. reshape并上采样到目标尺寸
    """
    
    def __init__(self, memory_bank, layer_idx_list, k_image=150,
                 use_positional_bank=True, pos_radius=1, device='cuda',
                 layer_weights=None, gaussian_kernel_size=5, gaussian_sigma=1.0):
        """
        参数:
            memory_bank: 记忆库字典，包含'cls'和'patch'
            layer_idx_list: 层索引列表
            k_image: 图像级最近邻数量
            use_positional_bank: 是否使用位置感知
            pos_radius: 位置感知半径
            device: 运行设备
            layer_weights: 每层权重，None表示均匀权重
            gaussian_kernel_size: 高斯核大小（官方默认: 5）
            gaussian_sigma: 高斯核标准差（官方默认: 1.0）
        """
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
            sigma=gaussian_sigma
        ).to(device)
    
    def detect(self, patch_features_list, cls_features_list=None, resize_mask=256):
        """
        检测异常
        
        参数:
            patch_features_list: 每层patch特征列表，每个元素为 [B, L, C] 或 (patch_tok, cls_tok)
            cls_features_list: 每层CLS特征列表
            resize_mask: 上采样后的尺寸
            
        返回:
            anomaly_map: 异常图 numpy array (resize_mask, resize_mask)
        """
        patch_list = []
        cls_list = []
        
        for feat in patch_features_list:
            if isinstance(feat, tuple):
                patch_list.append(feat[0])
                cls_list.append(feat[1])
            else:
                patch_list.append(feat)
        
        first_patch = patch_list[0]
        B = first_patch.shape[0]
        L = first_patch.shape[1]
        h = w = int(L ** 0.5)
        
        if len(cls_list) > 0:
            cls_x_global = cls_list[-1]
        else:
            raise ValueError("必须提供CLS特征用于图像级检索")
        
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
                    patch_scores_l = []
                    for j in range(L):
                        r = j // w
                        c = j % w
                        
                        r_min = max(0, r - self.pos_radius)
                        r_max = min(h - 1, r + self.pos_radius)
                        c_min = max(0, c - self.pos_radius)
                        c_max = min(w - 1, c + self.pos_radius)
                        
                        idx_list = []
                        for rr in range(r_min, r_max + 1):
                            base = rr * w
                            for cc in range(c_min, c_max + 1):
                                idx_list.append(base + cc)
                        
                        idx = torch.tensor(idx_list, device=self.device, dtype=torch.long)
                        neigh_local = neigh_feat[:, idx, :]
                        neigh_local = neigh_local.reshape(-1, neigh_local.shape[-1])
                        
                        qj = q_feat[j:j+1]
                        if neigh_local.numel() == 0:
                            patch_scores_l.append(torch.tensor(1.0, device=self.device))
                            continue
                        
                        sim = torch.matmul(qj, neigh_local.t())
                        nn_sim, _ = sim.max(dim=-1)
                        patch_scores_l.append(1.0 - nn_sim.squeeze(0))
                    
                    patch_score_l = torch.stack(patch_scores_l, dim=0)
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
            patch_maps,
            size=resize_mask,
            mode='bilinear',
            align_corners=False
        )
        
        anomaly_map = self.gaussian_kernel(anomaly_map)
        
        if B == 1:
            anomaly_map_np = anomaly_map[0, 0].detach().cpu().numpy()
        else:
            anomaly_map_np = anomaly_map.mean(dim=0)[0].detach().cpu().numpy()
        
        return anomaly_map_np