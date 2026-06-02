import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, DEFAULT_IMAGE_SIZE, DEFAULT_CROP_SIZE, LAYER_CONFIGS


class MVTecDataset(Dataset):
    """MVTec AD 数据集加载器"""
    
    def __init__(self, data_path, category, split='train', transform=None, 
                 gt_transform=None, image_size=DEFAULT_IMAGE_SIZE, crop_size=DEFAULT_CROP_SIZE):
        self.data_path = data_path
        self.category = category
        self.split = split
        
        if transform is None:
            mean_train = IMAGENET_MEAN
            std_train = IMAGENET_STD
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop(crop_size),
                transforms.Normalize(mean=mean_train, std=std_train)
            ])
        else:
            self.transform = transform
            
        if gt_transform is None:
            self.gt_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor()
            ])
        else:
            self.gt_transform = gt_transform
            
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
    
    def load_dataset(self):
        """加载数据集路径"""
        if self.split == 'train':
            img_path = os.path.join(self.data_path, self.category, 'train')
        else:
            img_path = os.path.join(self.data_path, self.category, 'test')
            gt_path_root = os.path.join(self.data_path, self.category, 'ground_truth')
        
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        
        defect_types = os.listdir(img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(img_path, defect_type, "*.png"))
                img_paths.extend(glob.glob(os.path.join(img_path, defect_type, "*.jpg")))
                img_paths.extend(glob.glob(os.path.join(img_path, defect_type, "*.jpeg")))
                img_paths = list(set(img_paths))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(img_path, defect_type, "*.png"))
                img_paths.extend(glob.glob(os.path.join(img_path, defect_type, "*.jpg")))
                img_paths.extend(glob.glob(os.path.join(img_path, defect_type, "*.jpeg")))
                img_paths = list(set(img_paths))
                gt_paths = glob.glob(os.path.join(gt_path_root, defect_type, "*.png"))
                gt_paths.extend(glob.glob(os.path.join(gt_path_root, defect_type, "*.jpg")))
                gt_paths.extend(glob.glob(os.path.join(gt_path_root, defect_type, "*.jpeg")))
                gt_paths = glob.glob(os.path.join(gt_path_root, defect_type, "*.png"))
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        if label == 0:
            gt_mask = torch.zeros([1, img.size(-2), img.size(-1)])
        else:
            gt_mask = Image.open(gt).convert('L')
            gt_mask = self.gt_transform(gt_mask)
        
        return img, gt_mask, label, img_path


class FeatureExtractor(torch.nn.Module):
    """DINOv3 特征提取器
    
    使用 get_intermediate_layers 同时获取 CLS token 和 Patch tokens
    支持官方 pth 和 ModelScope safetensors 格式自动检测
    """
    
    def __init__(self, model_name='dinov3_vitb16', pretrained_path=None, 
                 layer_idx_list=None, device='cuda'):
        super().__init__()
        if layer_idx_list is None:
            layer_idx_list = LAYER_CONFIGS.get(model_name, [3, 6, 9, 11])
        self.layer_idx_list = sorted(layer_idx_list)
        self.device = device
        
        import sys
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from .loader import load_model, detect_weight_format
        
        fmt = detect_weight_format(pretrained_path) if pretrained_path else 'official'
        if pretrained_path and fmt == 'modelscope':
            self.model = load_model(model_name=model_name, weight_path=pretrained_path, device=device)
        else:
            from .backbones_copy import load_dinov3_model
            if pretrained_path and fmt == 'official':
                self.model = load_dinov3_model(model_name, pretrained_weight_path=pretrained_path)
            else:
                self.model = load_dinov3_model(model_name)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        inter = self.model.get_intermediate_layers(
            x,
            n=self.layer_idx_list,
            return_class_token=True,
            reshape=False,
            norm=True,
        )
        
        patch_list = []
        cls_list = []
        for (patch_tok, cls_tok) in inter:
            patch_list.append(patch_tok)
            cls_list.append(cls_tok)
        
        return patch_list, cls_list


@torch.no_grad()
def build_memory_bank(encoder, dataset, device, batch_size=32,
                                layer_idx_list=None, num_workers=4):
    if layer_idx_list is None:
        layer_idx_list = [3, 6, 9, 11]
    """
    构建多层特征记忆库
    
    返回:
        layer_idx_list: 层索引列表
        cls_banks: CLS token 记忆库列表
        patch_banks: Patch token 记忆库列表
    """
    encoder.eval()
    layer_idx_list = sorted(layer_idx_list)
    num_layers = len(layer_idx_list)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=device.startswith("cuda"),
    )
    
    cls_feats = [[] for _ in range(num_layers)]
    patch_feats = [[] for _ in range(num_layers)]
    
    for step, (img, _, _, _) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        
        inter = encoder.get_intermediate_layers(
            img,
            n=layer_idx_list,
            return_class_token=True,
            reshape=False,
            norm=True,
        )
        
        for li, (patch_tok, cls_tok) in enumerate(inter):
            patch_feats[li].append(patch_tok.cpu())
            cls_feats[li].append(cls_tok.cpu())
        
        if step % 50 == 0:
            print(f"[构建记忆库] 步骤 {step}/{len(loader)}")
    
    cls_banks = [torch.cat(cls_feats[li], dim=0) for li in range(num_layers)]
    patch_banks = [torch.cat(patch_feats[li], dim=0) for li in range(num_layers)]
    
    return layer_idx_list, cls_banks, patch_banks


def get_gaussian_kernel(kernel_size=5, sigma=1.0):
    """创建高斯核用于异常图平滑 - 对齐官方参数"""
    import torch.nn as nn
    
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx = ax.repeat(kernel_size).view(kernel_size, -1)
    yy = ax.view(-1, 1).repeat(1, kernel_size)
    
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
    gaussian_filter = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
    gaussian_filter.weight.data = kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter