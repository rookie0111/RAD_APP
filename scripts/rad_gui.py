import sys
import os

_project_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_project_dir)
_root_dir = os.path.dirname(_parent_dir)
# RAD_APP must come before RAD_reproduction so local modelscope/ wins
for p in [_project_dir, _parent_dir, _root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)
# Add RAD_reproduction for dinov3 package (append to end to avoid shadowing modelscope)
_data_dir = os.path.join(_root_dir, 'RAD_reproduction')
if _data_dir not in sys.path:
    sys.path.append(_data_dir)

from modelscope.config import (
    IMAGENET_MEAN, IMAGENET_STD, WEIGHT_SEARCH_PATHS, ENCODER_WEIGHT_MAP,
    LAYER_CONFIGS, EMBED_DIMS, DEFAULT_IMAGE_SIZE, DEFAULT_CROP_SIZE,
    DEFAULT_K_IMAGE, DEFAULT_THRESHOLD, DEFAULT_BANK_PATH, DEFAULT_BATCH_SIZE,
    DEFAULT_POS_RADIUS, DEFAULT_USE_POSITIONAL_BANK,
)

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _BUNDLE_DIR = sys._MEIPASS
else:
    _BUNDLE_DIR = _parent_dir

def _w(path):
    if os.path.exists(path):
        return path
    bundled = os.path.join(_BUNDLE_DIR, os.path.relpath(path, _parent_dir))
    return bundled if os.path.exists(bundled) else path

_WEIGHT_SEARCH_PATHS = {k: [_w(p) for p in v] for k, v in WEIGHT_SEARCH_PATHS.items()}
_ENCODER_WEIGHT_MAP = {k: _w(v) for k, v in ENCODER_WEIGHT_MAP.items()}

import time
# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QTabWidget,
    QCheckBox, QLineEdit, QMessageBox, QSplitter, QSizePolicy,
    QFrame, QScrollArea, QGridLayout, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings
from PySide6.QtGui import QPixmap, QImage, QFont
import PySide6.QtGui as QtGui

import torch

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModernStyle:
    """现代化商务风格样式"""
    PRIMARY_COLOR = "#2E9AFE"
    SECONDARY_COLOR = "#58ACFA"
    ACCENT_COLOR = "#0080FF"
    BACKGROUND_COLOR = "#FAFAFA"
    CARD_COLOR = "#FFFFFF"
    TEXT_COLOR = "#2C3E50"
    TEXT_SECONDARY = "#7F8C8D"
    BORDER_COLOR = "#E0E0E0"
    SUCCESS_COLOR = "#27AE60"
    WARNING_COLOR = "#F39C12"
    ERROR_COLOR = "#E74C3C"
    
    @staticmethod
    def get_main_style():
        return f"""
        QMainWindow {{ background-color: {ModernStyle.BACKGROUND_COLOR}; }}
        QWidget {{ 
            background-color: {ModernStyle.BACKGROUND_COLOR}; 
            color: {ModernStyle.TEXT_COLOR};
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            font-size: 10pt;
        }}
        """
    
    @staticmethod
    def get_button_style():
        return f"""
        QPushButton {{
            background-color: {ModernStyle.PRIMARY_COLOR};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 24px;
            font-size: 10pt;
            font-weight: bold;
            min-height: 36px;
        }}
        QPushButton:hover {{ background-color: {ModernStyle.ACCENT_COLOR}; }}
        QPushButton:pressed {{ background-color: {ModernStyle.SECONDARY_COLOR}; }}
        QPushButton:disabled {{ background-color: {ModernStyle.TEXT_SECONDARY}; color: #CCCCCC; }}
        """
    
    @staticmethod
    def get_secondary_button_style():
        return f"""
        QPushButton {{
            background-color: transparent;
            color: {ModernStyle.PRIMARY_COLOR};
            border: 2px solid {ModernStyle.PRIMARY_COLOR};
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 10pt;
            font-weight: bold;
            min-height: 32px;
        }}
        QPushButton:hover {{
            background-color: {ModernStyle.PRIMARY_COLOR};
            color: white;
        }}
        """
    
    @staticmethod
    def get_group_style():
        return f"""
        QGroupBox {{
            background-color: {ModernStyle.CARD_COLOR};
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 8px;
            color: {ModernStyle.PRIMARY_COLOR};
        }}
        """
    
    @staticmethod
    def get_input_style():
        return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {ModernStyle.CARD_COLOR};
            border: 1px solid {ModernStyle.BORDER_COLOR};
            border-radius: 4px;
            padding: 6px 10px;
            min-height: 32px;
            font-size: 10pt;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {ModernStyle.PRIMARY_COLOR};
        }}
        QPushButton {{
            padding: 8px 16px;
            min-width: 80px;
            font-size: 10pt;
        }}
        QLabel {{
            font-size: 10pt;
            padding: 2px 4px;
        }}
        """
    
    @staticmethod
    def get_tab_style():
        return f"""
        QTabWidget::pane {{ border: none; background-color: {ModernStyle.BACKGROUND_COLOR}; }}
        QTabBar::tab {{
            background-color: transparent;
            color: {ModernStyle.TEXT_SECONDARY};
            border: none;
            border-bottom: 3px solid transparent;
            padding: 12px 24px;
            font-size: 11pt;
            font-weight: bold;
        }}
        QTabBar::tab:selected {{
            color: {ModernStyle.PRIMARY_COLOR};
            border-bottom: 3px solid {ModernStyle.PRIMARY_COLOR};
        }}
        """


class BuildBankThread(QThread):
    """构建记忆库后台线程"""
    progress = Signal(int)
    log = Signal(str)
    finished_signal = Signal(bool, str)
    
    def __init__(self, args_dict=None):
        super().__init__()
        self.args = args_dict if args_dict is not None else {}
    
    def run(self):
        try:
            from modelscope.dataset_v2 import MVTecDataset, build_memory_bank
            from torch.utils.data import ConcatDataset
            from torchvision import transforms
            
            device = self.args['device']
            self.log.emit(f"使用设备: {device}")
            
            # Load model - auto-detect format by inspecting weight keys
            from modelscope.loader import load_model as load_model_auto
            weight_path = self.args['encoder_weight']
            from modelscope.loader import detect_weight_format
            fmt = detect_weight_format(weight_path)
            self.log.emit(f"加载模型: {self.args['encoder_name']} ({fmt})")
            encoder = load_model_auto(
                model_name=self.args['encoder_name'],
                weight_path=weight_path,
                device=device
            )
            encoder.to(device)
            encoder.eval()
            
            # 准备数据
            train_data_list = []
            train_img_paths = []
            
            mean_train = IMAGENET_MEAN
            std_train = IMAGENET_STD
            data_transform = transforms.Compose([
                transforms.Resize((self.args['image_size'], self.args['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train)
            ])
            
            total_images = 0
            for category in self.args['categories']:
                self.log.emit(f"加载类别: {category}")
                train_dataset = MVTecDataset(
                    self.args['data_path'],
                    category,
                    split='train',
                    transform=data_transform,
                    image_size=self.args['image_size'],
                    crop_size=self.args['crop_size']
                )
                train_data_list.append(train_dataset)
                train_img_paths.extend(train_dataset.img_paths)
                total_images += len(train_dataset)
            
            self.log.emit(f"训练样本总数: {total_images}")
            
            # 合并数据集
            if len(train_data_list) > 1:
                train_dataset = ConcatDataset(train_data_list)
            else:
                train_dataset = train_data_list[0]
            
            # 构建记忆库
            self.log.emit("开始提取特征...")
            layer_idx_list = [int(x) for x in self.args['layers'].split()]
            
            layer_indices, cls_banks, patch_banks = build_memory_bank(
                encoder=encoder,
                dataset=train_dataset,
                device=device,
                batch_size=self.args['batch_size'],
                layer_idx_list=layer_idx_list,
                num_workers=0
            )
            
            # 保存记忆库
            os.makedirs(os.path.dirname(self.args['bank_path']) if os.path.dirname(self.args['bank_path']) else '.', exist_ok=True)
            torch.save({
                'layers': layer_indices,
                'cls_banks': cls_banks,
                'patch_banks': patch_banks,
                'train_img_paths': train_img_paths,
                'categories': self.args['categories']
            }, self.args['bank_path'])
            
            self.log.emit(f"记忆库已保存: {self.args['bank_path']}")
            self.finished_signal.emit(True, "记忆库构建成功")
            
        except Exception as e:
            self.log.emit(f"错误: {str(e)}")
            self.finished_signal.emit(False, str(e))


class DetectionThread(QThread):
    """检测后台线程"""
    log = Signal(str)
    finished_signal = Signal(dict)
    
    def __init__(self, args_dict):
        super().__init__()
        self.args = args_dict
        
    def _sync_device(self, device):
        if isinstance(device, str) and device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _save_inference_results(self, image_path, original_img, anomaly_map, save_dir, threshold=0.10):
        import cv2

        if not save_dir:
            save_dir = "./inference_results"
        os.makedirs(save_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        original_path = os.path.join(save_dir, f"{base_name}_{timestamp}_original.png")
        mask_path = os.path.join(save_dir, f"{base_name}_{timestamp}_mask.png")
        result_path = os.path.join(save_dir, f"{base_name}_{timestamp}_result.png")

        # 原图尺寸
        orig_w, orig_h = original_img.size

        # 保存原图
        original_img.save(original_path)

        # anomaly_map 缩放回原图尺寸
        anomaly_map_resized = cv2.resize(
            anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # 保存 mask
        mask = (anomaly_map_resized >= threshold).astype(np.uint8) * 255
        Image.fromarray(mask, mode='L').save(mask_path)

        # 保存检测结果图（原图 + 异常区域叠加）
        original_np = np.array(original_img).copy()

        vis_map = anomaly_map_resized.copy()
        vis_map[vis_map < threshold] = 0  # 小于阈值的不显示

        # 映射到伪彩色
        vis_uint8 = np.clip(vis_map / 0.3 * 255.0, 0, 255).astype(np.uint8)
        color_bgr = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        result_img = original_np.copy()
        alpha = 0.5
        valid = vis_map > 0
        result_img[valid] = (
            (1 - alpha) * original_np[valid] + alpha * color_rgb[valid]
        ).astype(np.uint8)

        Image.fromarray(result_img).save(result_path)

        return {
            'save_dir': save_dir,
            'original_path': original_path,
            'mask_path': mask_path,
            'result_path': result_path
        }

          
    def run(self):
        try:
            from modelscope.rad_detector import RADDetector
            import cv2
            
            device = self.args['device']
            self.log.emit(f"使用设备: {device}")
            
            # 加载记忆库
            self.log.emit("加载记忆库...")
            bank_data = torch.load(self.args['bank_path'], map_location='cpu')
            layer_indices = bank_data['layers']
            cls_banks = bank_data['cls_banks']
            patch_banks = bank_data['patch_banks']
            
            # 获取记忆库的维度信息来确定编码器类型
            cls_dim = cls_banks[-1].shape[-1]  # 最高层CLS特征维度
            self.log.emit(f"记忆库CLS维度: {cls_dim}")
            
            # 根据记忆库维度自动选择正确的编码器
            if cls_dim == 768:
                encoder_name = 'dinov3_vitb16'
                self.log.emit("检测到ViT-Base记忆库，使用dinov3_vitb16")
            elif cls_dim == 384:
                encoder_name = 'dinov3_vits16'
                self.log.emit("检测到ViT-Small记忆库，使用dinov3_vits16")
            elif cls_dim == 1024:
                encoder_name = 'dinov3_vitl16'
                self.log.emit("检测到ViT-Large记忆库，使用dinov3_vitl16")
            else:
                # 使用用户指定的编码器
                encoder_name = self.args['encoder_name']
                self.log.emit(f"未知维度{cls_dim}，使用用户指定编码器: {encoder_name}")
            
            # 根据编码器类型选择权重路径，自动检测格式
            weight_search_paths = _WEIGHT_SEARCH_PATHS
            
            weight_path = None
            for p in weight_search_paths.get(encoder_name, []):
                if os.path.exists(p):
                    weight_path = p
                    break
            
            if weight_path is None:
                raise FileNotFoundError(f"找不到 {encoder_name} 的权重文件")
            
            # 加载模型 - 自动检测权重格式
            from modelscope.loader import load_model as load_model_auto, detect_weight_format
            fmt = detect_weight_format(weight_path)
            self.log.emit(f"加载DINOv3模型: {encoder_name} (格式: {fmt})")
            encoder = load_model_auto(model_name=encoder_name, weight_path=weight_path, device=device)
            encoder.to(device)
            encoder.eval()
            
            # 创建检测器
            self.log.emit("初始化检测器...")
            detector = RADDetector(
                memory_bank={'cls': cls_banks, 'patch': patch_banks},
                layer_idx_list=layer_indices,
                k_image=self.args['k_image'],
                use_positional_bank=self.args['use_positional_bank'],
                pos_radius=self.args['pos_radius'],
                device=device
            )
            
            # 加载并预处理图像
            self.log.emit("处理图像...")
            mean_train = IMAGENET_MEAN
            std_train = IMAGENET_STD
            
            # 加载原始图像
            original_img = Image.open(self.args['image_path']).convert('RGB')
            original_w, original_h = original_img.size
            
            # 预处理：直接Resize到512x512，不做裁剪
            img_resized = original_img.resize((512, 512))
            
            # 转换为tensor并归一化
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=mean_train, std=std_train)
            img_tensor = normalize(to_tensor(img_resized)).unsqueeze(0).to(device)
            
            # ---------------------------
            # Warmup
            # ---------------------------
            self.log.emit("Warmup开始...")
            self._sync_device(device)
            warmup_start = time.perf_counter()

            with torch.no_grad():
                warmup_inter = encoder.get_intermediate_layers(
                    img_tensor,
                    n=layer_indices,
                    return_class_token=True,
                    reshape=False,
                    norm=True,
                )
                _ = detector.detect(warmup_inter)

            self._sync_device(device)
            warmup_end = time.perf_counter()
            warmup_time_ms = (warmup_end - warmup_start) * 1000.0
            self.log.emit(f"Warmup时间: {warmup_time_ms:.2f} ms")

            # ---------------------------
            # 正式单张推理计时
            # ---------------------------
            self.log.emit("开始正式推理...")
            self._sync_device(device)
            infer_start = time.perf_counter()

            with torch.no_grad():
                inter = encoder.get_intermediate_layers(
                    img_tensor,
                    n=layer_indices,
                    return_class_token=True,
                    reshape=False,
                    norm=True,
                )
                anomaly_map = detector.detect(inter)  # 输出256x256

            self._sync_device(device)
            infer_end = time.perf_counter()
            inference_time_ms = (infer_end - infer_start) * 1000.0
            self.log.emit(f"单张推理时间: {inference_time_ms:.2f} ms")

            
            # 计算异常分数
            anomaly_score = float(np.max(anomaly_map))

            # 上采样异常图到512x512（用于界面显示）
            anomaly_map_512 = cv2.resize(anomaly_map, (512, 512))

            # 图像用于显示
            img_resized_array = np.ascontiguousarray(np.array(img_resized))

            # 保存推理结果
            threshold = float(self.args.get('threshold', 0.10))
            save_dir = self.args.get('save_dir', './inference_results')
            # saved_paths = self._save_inference_results(
            #     image_path=self.args['image_path'],
            #     original_img=original_img,
            #     anomaly_map=anomaly_map,
            #     save_dir=save_dir,
            #     threshold=threshold
            # )

            # self.log.emit(f"推理结果已保存到: {saved_paths['save_dir']}")
            # self.log.emit(f"原图: {saved_paths['original_path']}")
            # self.log.emit(f"Mask: {saved_paths['mask_path']}")
            # self.log.emit(f"检测结果图: {saved_paths['result_path']}")

            # 生成结果字典
            result = {
                'original_image': img_resized_array,
                'anomaly_map': anomaly_map_512,
                'anomaly_score': anomaly_score,
                'has_anomaly': anomaly_score > 0.10,
                'success': True,
                'is_cropped': False,
                'original_size': (original_h, original_w),
                'warmup_time_ms': warmup_time_ms,
                'inference_time_ms': inference_time_ms,
                # 'saved_paths': saved_paths
            }

            
            self.finished_signal.emit(result)
            
        except Exception as e:
            self.log.emit(f"检测错误: {str(e)}")
            self.finished_signal.emit({'success': False, 'error': str(e)})


class EvaluationThread(QThread):
    """评估后台线程"""
    log = Signal(str)
    progress = Signal(int)
    finished_signal = Signal(dict)
    
    def __init__(self, args_dict):
        super().__init__()
        self.args = args_dict
    
    def run(self):
        try:
            from modelscope.dataset_v2 import MVTecDataset
            from modelscope.rad_detector import RADDetector
            from torchvision import transforms
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            device = self.args['device']
            self.log.emit(f"使用设备: {device}")
            
            # 加载记忆库和模型
            self.log.emit("加载记忆库...")
            bank_data = torch.load(self.args['bank_path'], map_location='cpu')
            layer_indices = bank_data['layers']
            cls_banks = bank_data['cls_banks']
            patch_banks = bank_data['patch_banks']
            
            # 根据记忆库维度自动选择正确的编码器
            cls_dim = cls_banks[-1].shape[-1]
            self.log.emit(f"记忆库CLS维度: {cls_dim}")
            
            # 根据记忆库维度自动选择正确的编码器
            cls_dim = cls_banks[-1].shape[-1]
            self.log.emit(f"记忆库CLS维度: {cls_dim}")
            
            if cls_dim == 768:
                encoder_name = 'dinov3_vitb16'
                self.log.emit("检测到ViT-Base记忆库")
            elif cls_dim == 384:
                encoder_name = 'dinov3_vits16'
                self.log.emit("检测到ViT-Small记忆库")
            elif cls_dim == 1024:
                encoder_name = 'dinov3_vitl16'
                self.log.emit("检测到ViT-Large记忆库")
            else:
                encoder_name = self.args.get('encoder_name', 'dinov3_vitb16')
                self.log.emit(f"未知维度{cls_dim}，使用编码器: {encoder_name}")
            
            # 根据编码器类型选择权重路径，自动检测格式
            weight_search_paths = _WEIGHT_SEARCH_PATHS
            
            weight_path = None
            for p in weight_search_paths.get(encoder_name, []):
                if os.path.exists(p):
                    weight_path = p
                    break
            
            if weight_path is None:
                raise FileNotFoundError(f"找不到 {encoder_name} 的权重文件")
            
            # 加载模型 - 自动检测权重格式
            from modelscope.loader import load_model as load_model_auto, detect_weight_format
            fmt = detect_weight_format(weight_path)
            self.log.emit(f"加载模型: {encoder_name} (格式: {fmt})")
            encoder = load_model_auto(model_name=encoder_name, weight_path=weight_path, device=device)
            encoder.to(device)
            encoder.eval()
            
            detector = RADDetector(
                memory_bank={'cls': cls_banks, 'patch': patch_banks},
                layer_idx_list=layer_indices,
                k_image=self.args['k_image'],
                use_positional_bank=self.args['use_positional_bank'],
                pos_radius=self.args['pos_radius'],
                device=device
            )
            
            # 加载测试数据
            mean_train = IMAGENET_MEAN
            std_train = IMAGENET_STD
            data_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train)
            ])
            gt_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
            
            test_dataset = MVTecDataset(
                self.args['data_path'],
                self.args['category'],
                split='test',
                transform=data_transform,
                gt_transform=gt_transform,
                image_size=512,
                crop_size=512
            )
            
            self.log.emit(f"测试样本数: {len(test_dataset)}")
            
            # 收集结果
            gt_list = []
            pred_list = []
            
            for i in range(len(test_dataset)):
                img, gt, label, _ = test_dataset[i]
                
                # 提取特征
                img_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    inter = encoder.get_intermediate_layers(
                        img_tensor,
                        n=layer_indices,
                        return_class_token=True,
                        reshape=False,
                        norm=True,
                    )
                
                # 检测
                anomaly_map = detector.detect(inter)
                anomaly_score = float(np.max(anomaly_map))
                
                gt_list.append(label)
                pred_list.append(anomaly_score)
                
                if i % 10 == 0:
                    self.progress.emit(int((i / len(test_dataset)) * 100))
                    self.log.emit(f"处理进度: {i}/{len(test_dataset)}")
            
            self.progress.emit(100)
            
            # 计算指标
            gt_array = np.array(gt_list)
            pred_array = np.array(pred_list)
            
            auroc = roc_auc_score(gt_array, pred_array)
            ap = average_precision_score(gt_array, pred_array)
            
            # 计算最佳F1
            from sklearn.metrics import f1_score
            best_f1 = 0
            best_thresh = 0
            for thresh in np.linspace(0, 1, 100):
                pred_labels = (pred_array > thresh).astype(int)
                if pred_labels.sum() > 0:
                    f1 = f1_score(gt_array, pred_labels)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
            
            result = {
                'auroc': auroc,
                'ap': ap,
                'best_f1': best_f1,
                'best_threshold': best_thresh,
                'total_samples': len(test_dataset),
                'anomaly_samples': int(gt_array.sum()),
                'success': True
            }
            
            self.finished_signal.emit(result)
            
        except Exception as e:
            self.log.emit(f"评估错误: {str(e)}")
            self.finished_signal.emit({'success': False, 'error': str(e)})


class ExportONNXThread(QThread):
    log = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, args_dict):
        super().__init__()
        self.args = args_dict

    def run(self):
        try:
            from modelscope.onnx_export import (
                load_encoder, export_dinov3_onnx, export_rad_onnx,
                export_full_pipeline_onnx
            )

            mode = self.args['mode']
            device = self.args['device']
            output_dir = self.args['output_dir']
            output_name = self.args.get('output_name', 'rad_pipeline_full')
            os.makedirs(output_dir, exist_ok=True)

            if mode in ('full',):
                onnx_path = os.path.join(output_dir, f'{output_name}.onnx')
            elif mode == 'dinov3':
                onnx_path = os.path.join(output_dir, f'{output_name}_encoder.onnx')
            elif mode == 'rad':
                onnx_path = os.path.join(output_dir, f'{output_name}_detector.onnx')
            elif mode in ('separated', 'all'):
                onnx_path = None
            else:
                onnx_path = os.path.join(output_dir, f'{output_name}.onnx')

            encoder = None
            if mode in ['dinov3', 'full', 'separated', 'all']:
                self.log.emit(f"加载编码器: {self.args['encoder_name']}")
                encoder = load_encoder(
                    self.args['encoder_name'],
                    self.args['encoder_weight'],
                    device=device
                )

            success = True

            if mode == 'dinov3':
                self.log.emit("导出 DINOv3 编码器 ONNX...")
                success = export_dinov3_onnx(
                    encoder=encoder,
                    output_path=onnx_path,
                    layer_indices=self.args['layer_idx_list'],
                    img_size=self.args['img_size'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
            elif mode == 'rad':
                self.log.emit("导出 RAD 检测器 ONNX...")
                success = export_rad_onnx(
                    memory_bank_path=self.args['bank_path'],
                    output_path=onnx_path,
                    k_image=self.args['k_image'],
                    use_positional_bank=self.args['use_positional_bank'],
                    pos_radius=self.args['pos_radius'],
                    resize_mask=self.args['resize_mask'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
            elif mode == 'full':
                self.log.emit("导出完整 Pipeline ONNX...")
                success = export_full_pipeline_onnx(
                    encoder=encoder,
                    memory_bank_path=self.args['bank_path'],
                    output_path=onnx_path,
                    layer_indices=self.args['layer_idx_list'],
                    img_size=self.args['img_size'],
                    k_image=self.args['k_image'],
                    use_positional_bank=self.args['use_positional_bank'],
                    pos_radius=self.args['pos_radius'],
                    resize_mask=self.args['resize_mask'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                    low_memory=self.args.get('low_memory', False),
                )
            elif mode == 'separated':
                enc_path = os.path.join(output_dir, f'{output_name}_encoder.onnx')
                det_path = os.path.join(output_dir, f'{output_name}_detector.onnx')
                self.log.emit("导出 DINOv3 编码器 ONNX...")
                success &= export_dinov3_onnx(
                    encoder=encoder,
                    output_path=enc_path,
                    layer_indices=self.args['layer_idx_list'],
                    img_size=self.args['img_size'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
                self.log.emit("导出 RAD 检测器 ONNX...")
                success &= export_rad_onnx(
                    memory_bank_path=self.args['bank_path'],
                    output_path=det_path,
                    k_image=self.args['k_image'],
                    use_positional_bank=self.args['use_positional_bank'],
                    pos_radius=self.args['pos_radius'],
                    resize_mask=self.args['resize_mask'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
            elif mode == 'all':
                enc_path = os.path.join(output_dir, f'{output_name}_encoder.onnx')
                det_path = os.path.join(output_dir, f'{output_name}_detector.onnx')
                full_path = os.path.join(output_dir, f'{output_name}.onnx')
                self.log.emit("导出 DINOv3 编码器 ONNX...")
                success &= export_dinov3_onnx(
                    encoder=encoder,
                    output_path=enc_path,
                    layer_indices=self.args['layer_idx_list'],
                    img_size=self.args['img_size'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
                self.log.emit("导出 RAD 检测器 ONNX...")
                success &= export_rad_onnx(
                    memory_bank_path=self.args['bank_path'],
                    output_path=det_path,
                    k_image=self.args['k_image'],
                    use_positional_bank=self.args['use_positional_bank'],
                    pos_radius=self.args['pos_radius'],
                    resize_mask=self.args['resize_mask'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                )
                self.log.emit("导出完整 Pipeline ONNX（需要较多内存）...")
                success &= export_full_pipeline_onnx(
                    encoder=encoder,
                    memory_bank_path=self.args['bank_path'],
                    output_path=full_path,
                    layer_indices=self.args['layer_idx_list'],
                    img_size=self.args['img_size'],
                    k_image=self.args['k_image'],
                    use_positional_bank=self.args['use_positional_bank'],
                    pos_radius=self.args['pos_radius'],
                    resize_mask=self.args['resize_mask'],
                    opset_version=self.args['opset_version'],
                    verify=self.args['verify'],
                    device=device,
                    low_memory=self.args.get('low_memory', False),
                )

            if success:
                self.finished_signal.emit(True, f"ONNX 导出成功\n输出目录: {output_dir}")
            else:
                self.finished_signal.emit(False, "ONNX 导出失败，请检查日志")
        except Exception as e:
            import traceback
            self.log.emit(f"导出错误: {str(e)}")
            self.log.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


class RAD_GUI(QMainWindow):
    """RAD异常检测系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LinkVision异常检测系统")
        self.setMinimumSize(960, 720)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory_bank = None
        self.encoder = None
        self.current_image_path = None
        self.current_result = None
        self.current_anomaly_map = None
        self.detect_thread = None
        
        settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "params.ini")
        self.settings = QSettings(settings_path, QSettings.IniFormat)
        
        self.setup_ui()
        self.apply_styles()
        self.load_settings()
        self.log("RAD 异常检测系统已启动")
        self.log("当前配置：训练 / 评估 / 推理统一采用 512x512，不进行裁剪")
        
        # 初始最大化显示
        self.showMaximized()
        self.log(f"使用设备: {self.device}")
    
    def apply_styles(self):
        """应用现代化样式"""
        self.setStyleSheet(
            ModernStyle.get_main_style() +
            ModernStyle.get_button_style() +
            ModernStyle.get_group_style() +
            ModernStyle.get_input_style() +
            ModernStyle.get_tab_style()
        )
    
    def setup_ui(self):
        """设置UI界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        left_widget = self.create_left_panel()
        main_layout.addWidget(left_widget, stretch=3)
        
        right_widget = self.create_right_panel()
        main_layout.addWidget(right_widget, stretch=1)
    
    def create_left_panel(self):
        """创建左侧面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title_frame = self.create_title_bar()
        layout.addWidget(title_frame)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_bank_tab(), "  特征记忆库  ")
        self.tabs.addTab(self.create_eval_tab(), "  性能评估  ")
        self.tabs.addTab(self.create_single_test_tab(), "  单样本检测  ")
        self.tabs.addTab(self.create_onnx_export_tab(), "  ONNX 导出  ")
        layout.addWidget(self.tabs, stretch=1)
        
        return widget
    
    def create_title_bar(self):
        """创建标题栏"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {ModernStyle.CARD_COLOR};
                border-radius: 8px;
                border: 1px solid {ModernStyle.BORDER_COLOR};
            }}
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(15, 10, 15, 10)
        
        title = QLabel("LinkVision 异常检测系统")
        title.setStyleSheet(f"""
            font-size: 20pt;
            font-weight: bold;
            color: {ModernStyle.PRIMARY_COLOR};
        """)
        layout.addWidget(title)
        layout.addStretch()
        
        device_label = QLabel(f"设备: {self.device.upper()}")
        device_label.setStyleSheet(f"""
            font-size: 9pt;
            color: {ModernStyle.TEXT_SECONDARY};
            padding: 4px 10px;
            background-color: {ModernStyle.BACKGROUND_COLOR};
            border-radius: 4px;
        """)
        layout.addWidget(device_label)
        
        return frame
    
    def create_right_panel(self):
        """创建右侧面板（日志区域）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        log_header = QLabel("运行日志")
        log_header.setStyleSheet(f"""
            font-size: 12pt;
            font-weight: bold;
            color: {ModernStyle.TEXT_COLOR};
            padding: 6px 0;
        """)
        layout.addWidget(log_header)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumWidth(320)
        layout.addWidget(self.log_text, stretch=1)
        
        clear_btn = QPushButton("清空日志")
        clear_btn.setStyleSheet(ModernStyle.get_secondary_button_style())
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
        
        return widget
    
    def create_styled_group(self, title):
        """创建样式化的分组框"""
        group = QGroupBox(title)
        group.setStyleSheet(ModernStyle.get_group_style())
        return group
    
    def create_bank_tab(self):
        """创建记忆库构建标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)
        
        # 数据路径组
        path_group = self.create_styled_group("数据路径")
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("数据集路径:"), stretch=0)
        self.data_path_input = QLineEdit("./data/mvtec")
        data_btn = QPushButton("浏览")
        data_btn.setMaximumWidth(80)
        data_btn.clicked.connect(lambda: self.browse_folder(self.data_path_input))
        data_layout.addWidget(self.data_path_input, stretch=1)
        data_layout.addWidget(data_btn, stretch=0)
        path_layout.addLayout(data_layout)
        
        bank_layout = QHBoxLayout()
        bank_layout.addWidget(QLabel("记忆库保存:"), stretch=0)
        self.bank_path_input = QLineEdit("./bank/memory_bank_official.pth")
        bank_btn = QPushButton("浏览")
        bank_btn.setMaximumWidth(80)
        bank_btn.clicked.connect(lambda: self.save_file_dialog(self.bank_path_input))
        bank_layout.addWidget(self.bank_path_input, stretch=1)
        bank_layout.addWidget(bank_btn, stretch=0)
        path_layout.addLayout(bank_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 类别设置组
        category_group = self.create_styled_group("类别设置")
        cat_layout = QVBoxLayout()
        cat_layout.addWidget(QLabel("类别名称 (多个用空格分隔):"))
        self.category_input = QLineEdit("metal_nut")
        self.category_input.setPlaceholderText("例如: metal_nut bottle cable")
        cat_layout.addWidget(self.category_input)
        category_group.setLayout(cat_layout)
        layout.addWidget(category_group)
        
        # 模型设置组
        model_group = self.create_styled_group("模型设置")
        model_grid = QGridLayout()
        model_grid.setSpacing(12)
        
        model_grid.addWidget(QLabel("编码器:"), 0, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(["dinov3_vitb16", "dinov3_vits16", "dinov3_vitl16"])
        model_grid.addWidget(self.encoder_combo, 0, 1)
        
        # 模型名称到权重文件的映射
        self.encoder_weight_map = _ENCODER_WEIGHT_MAP
        
        model_grid.addWidget(QLabel("权重路径:"), 1, 0)
        weight_layout = QHBoxLayout()
        self.weight_path_input = QLineEdit(_ENCODER_WEIGHT_MAP.get('dinov3_vitb16', ''))
        
        # 连接编码器选择变化信号，自动更新权重路径
        self.encoder_combo.currentTextChanged.connect(self.on_encoder_changed)
        weight_btn = QPushButton("浏览")
        weight_btn.setMaximumWidth(80)
        weight_btn.clicked.connect(lambda: self.open_file_dialog(self.weight_path_input))
        weight_layout.addWidget(self.weight_path_input)
        weight_layout.addWidget(weight_btn)
        model_grid.addLayout(weight_layout, 1, 1)
        
        model_grid.addWidget(QLabel("提取层:"), 2, 0)
        self.layer_input = QLineEdit("3 6 9 11")
        model_grid.addWidget(self.layer_input, 2, 1)
        
        model_group.setLayout(model_grid)
        layout.addWidget(model_group)
        
        # 训练参数组
        train_group = self.create_styled_group("训练参数")
        train_grid = QGridLayout()
        train_grid.setSpacing(12)
        
        train_grid.addWidget(QLabel("Batch Size:"), 0, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        train_grid.addWidget(self.batch_spin, 0, 1)
        
        train_grid.addWidget(QLabel("图像尺寸:"), 0, 2)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(128, 1024)
        self.image_size_spin.setValue(512)
        train_grid.addWidget(self.image_size_spin, 0, 3)
        
        train_group.setLayout(train_grid)
        layout.addWidget(train_group)

        # 构建按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        build_btn = QPushButton("开始构建记忆库")
        build_btn.setMinimumWidth(180)
        build_btn.setMinimumHeight(40)
        build_btn.clicked.connect(self.build_memory_bank)
        btn_layout.addWidget(build_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.bank_progress = QProgressBar()
        self.bank_progress.setMaximumHeight(6)
        layout.addWidget(self.bank_progress)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll
    
    def create_eval_tab(self):
        """创建性能评估标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)
        
        path_group = self.create_styled_group("路径设置")
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)
        
        bank_layout = QHBoxLayout()
        bank_layout.addWidget(QLabel("记忆库:"), stretch=0)
        self.eval_bank_path = QLineEdit("./bank/memory_bank_official.pth")
        bank_btn = QPushButton("浏览")
        bank_btn.setMaximumWidth(80)
        bank_btn.clicked.connect(lambda: self.open_file_dialog(self.eval_bank_path))
        bank_layout.addWidget(self.eval_bank_path, stretch=1)
        bank_layout.addWidget(bank_btn, stretch=0)
        path_layout.addLayout(bank_layout)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("数据集:"), stretch=0)
        self.eval_data_path = QLineEdit("./data/mvtec")
        data_btn = QPushButton("浏览")
        data_btn.setMaximumWidth(80)
        data_btn.clicked.connect(lambda: self.browse_folder(self.eval_data_path))
        data_layout.addWidget(self.eval_data_path, stretch=1)
        data_layout.addWidget(data_btn, stretch=0)
        path_layout.addLayout(data_layout)
        
        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("类别:"), stretch=0)
        self.eval_category = QLineEdit("metal_nut")
        cat_layout.addWidget(self.eval_category, stretch=1)
        path_layout.addLayout(cat_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        param_group = self.create_styled_group("检测参数")
        param_grid = QGridLayout()
        param_grid.setSpacing(12)
        
        param_grid.addWidget(QLabel("K值:"), 0, 0)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 500)
        self.k_spin.setValue(30)
        param_grid.addWidget(self.k_spin, 0, 1)
        
        param_grid.addWidget(QLabel("邻域半径:"), 0, 2)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(0, 5)
        self.radius_spin.setValue(1)
        param_grid.addWidget(self.radius_spin, 0, 3)
        
        self.pos_check = QCheckBox("启用位置感知检索")
        self.pos_check.setChecked(True)
        param_grid.addWidget(self.pos_check, 1, 0, 1, 4)
        
        param_group.setLayout(param_grid)
        layout.addWidget(param_group)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        eval_btn = QPushButton("开始性能评估")
        eval_btn.setMinimumWidth(180)
        eval_btn.setMinimumHeight(40)
        eval_btn.clicked.connect(self.run_evaluation)
        btn_layout.addWidget(eval_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.eval_progress = QProgressBar()
        self.eval_progress.setMaximumHeight(6)
        layout.addWidget(self.eval_progress)
        
        result_group = self.create_styled_group("评估结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(180)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        scroll.setWidget(widget)
        return scroll
    
    def create_single_test_tab(self):
        """创建单样本测试标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 控制区域
        top_splitter = QSplitter(Qt.Horizontal)
        
        img_group = self.create_styled_group("图像选择")
        img_layout = QVBoxLayout()
        img_layout.setSpacing(10)
        
        img_path_layout = QHBoxLayout()
        img_path_layout.addWidget(QLabel("测试图像:"), stretch=0)
        self.test_img_path = QLineEdit()
        img_btn = QPushButton("选择图像")
        img_btn.setMinimumWidth(80)
        img_btn.clicked.connect(self.select_test_image)
        img_path_layout.addWidget(self.test_img_path, stretch=1)
        img_path_layout.addWidget(img_btn, stretch=0)
        img_layout.addLayout(img_path_layout)
        
        bank_layout = QHBoxLayout()
        bank_layout.addWidget(QLabel("记忆库:"), stretch=0)
        self.test_bank_path = QLineEdit("./bank/memory_bank_official.pth")
        bank_btn = QPushButton("浏览文件")
        bank_btn.setMinimumWidth(80)
        bank_btn.clicked.connect(lambda: self.open_file_dialog(self.test_bank_path))
        bank_layout.addWidget(self.test_bank_path, stretch=1)
        bank_layout.addWidget(bank_btn, stretch=0)
        img_layout.addLayout(bank_layout)
        
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("保存目录:"), stretch=0)
        self.save_dir_input = QLineEdit("./inference_results")
        save_btn = QPushButton("选择文件夹")
        save_btn.setMinimumWidth(80)
        save_btn.clicked.connect(lambda: self.browse_folder(self.save_dir_input))
        save_layout.addWidget(self.save_dir_input, stretch=1)
        save_layout.addWidget(save_btn, stretch=0)
        img_layout.addLayout(save_layout)


        img_group.setLayout(img_layout)
        top_splitter.addWidget(img_group)
        
        param_group = self.create_styled_group("检测参数")
        param_grid = QGridLayout()
        param_grid.setSpacing(10)
        
        param_grid.addWidget(QLabel("K值:"), 0, 0)
        self.test_k_spin = QSpinBox()
        self.test_k_spin.setRange(1, 100)
        self.test_k_spin.setValue(30)
        param_grid.addWidget(self.test_k_spin, 0, 1)
        
        param_grid.addWidget(QLabel("半径:"), 0, 2)
        self.test_radius_spin = QSpinBox()
        self.test_radius_spin.setRange(0, 5)
        self.test_radius_spin.setValue(1)
        param_grid.addWidget(self.test_radius_spin, 0, 3)
        
        self.test_pos_check = QCheckBox("位置感知")
        self.test_pos_check.setChecked(True)
        param_grid.addWidget(self.test_pos_check, 1, 0)
        
        self.fast_mode_check = QCheckBox("快速模式")
        self.fast_mode_check.setChecked(True)
        self.fast_mode_check.setToolTip("使用矩阵化计算加速检测")
        param_grid.addWidget(self.fast_mode_check, 1, 1)
        
        detect_btn = QPushButton("开始检测")
        detect_btn.setMinimumHeight(40)
        detect_btn.setMinimumWidth(120)
        detect_btn.clicked.connect(self.run_single_test)
        param_grid.addWidget(detect_btn, 1, 2, 1, 2)
        
        # 异常分数阈值滑动条
        param_grid.addWidget(QLabel("阈值:"), 2, 0)
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 30)  # 0.00 - 0.30
        self.threshold_slider.setValue(10)     # 默认 0.10
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.10")
        self.threshold_label.setFixedWidth(40)
        self.threshold_label.setStyleSheet("font-weight: bold; color: #2E9AFE;")
        threshold_layout.addWidget(self.threshold_label)
        param_grid.addLayout(threshold_layout, 2, 1, 1, 3)
        
        threshold_hint = QLabel("推荐: 0.08-0.15")
        threshold_hint.setStyleSheet("color: #7F8C8D; font-size: 9pt;")
        param_grid.addWidget(threshold_hint, 3, 0, 1, 4)
        
        param_group.setLayout(param_grid)
        top_splitter.addWidget(param_group)
        
        top_splitter.setSizes([300, 200])
        layout.addWidget(top_splitter, stretch=0)
        
        # 结果显示区域
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        self.original_frame = self.create_image_frame("原始图像")
        bottom_splitter.addWidget(self.original_frame)
        
        self.anomaly_frame = self.create_image_frame("异常热力图")
        bottom_splitter.addWidget(self.anomaly_frame)
        
        self.overlay_frame = self.create_image_frame("检测结果")
        bottom_splitter.addWidget(self.overlay_frame)
        
        bottom_splitter.setSizes([250, 250, 250])
        layout.addWidget(bottom_splitter, stretch=1)
        
        self.single_result_text = QTextEdit()
        self.single_result_text.setReadOnly(True)
        self.single_result_text.setMaximumHeight(100)
        layout.addWidget(self.single_result_text, stretch=0)
        
        return widget

    def create_onnx_export_tab(self):
        """创建ONNX导出标签页"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignTop)

        # 导出模式
        mode_group = self.create_styled_group("导出模式")
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)
        self.onnx_mode_combo = QComboBox()
        self.onnx_mode_combo.addItems(['full', 'separated', 'dinov3', 'rad', 'all'])
        self.onnx_mode_combo.setCurrentText('separated')
        self.onnx_mode_combo.currentTextChanged.connect(self.on_onnx_mode_changed)
        mode_form = QGridLayout()
        mode_form.addWidget(QLabel("导出模式:"), 0, 0)
        mode_form.addWidget(self.onnx_mode_combo, 0, 1)
        mode_hint = QLabel("full=完整pipeline  separated=编码器+检测器(推荐)  dinov3=仅编码器  rad=仅检测器  all=全部")
        mode_hint.setStyleSheet(f"color: {ModernStyle.TEXT_SECONDARY}; font-size: 9pt;")
        mode_form.addWidget(mode_hint, 1, 0, 1, 2)
        mode_layout.addLayout(mode_form)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # 模型设置
        model_group = self.create_styled_group("模型设置")
        model_grid = QGridLayout()
        model_grid.setSpacing(12)
        model_grid.addWidget(QLabel("编码器:"), 0, 0)
        self.onnx_encoder_combo = QComboBox()
        self.onnx_encoder_combo.addItems(["dinov3_vitb16", "dinov3_vits16", "dinov3_vitl16"])
        self.onnx_encoder_combo.currentTextChanged.connect(self.on_onnx_encoder_changed)
        model_grid.addWidget(self.onnx_encoder_combo, 0, 1)
        model_grid.addWidget(QLabel("权重路径:"), 1, 0)
        weight_layout = QHBoxLayout()
        self.onnx_weight_input = QLineEdit(_ENCODER_WEIGHT_MAP.get('dinov3_vitb16', ''))
        weight_btn = QPushButton("浏览")
        weight_btn.setMaximumWidth(80)
        weight_btn.clicked.connect(lambda: self.open_file_dialog(self.onnx_weight_input))
        weight_layout.addWidget(self.onnx_weight_input)
        weight_layout.addWidget(weight_btn)
        model_grid.addLayout(weight_layout, 1, 1)
        model_grid.addWidget(QLabel("提取层:"), 2, 0)
        self.onnx_layer_input = QLineEdit("3 6 9 11")
        self.onnx_layer_input.setPlaceholderText("空格分隔，如: 3 6 9 11 或 5 11 17 23")
        model_grid.addWidget(self.onnx_layer_input, 2, 1)
        model_group.setLayout(model_grid)
        layout.addWidget(model_group)

        # 记忆库设置
        bank_group = self.create_styled_group("记忆库设置")
        bank_layout = QGridLayout()
        bank_layout.setSpacing(12)
        bank_layout.addWidget(QLabel("记忆库:"), 0, 0)
        bank_file_layout = QHBoxLayout()
        self.onnx_bank_input = QLineEdit("./bank/memory_bank_official.pth")
        bank_btn = QPushButton("浏览")
        bank_btn.setMaximumWidth(80)
        bank_btn.clicked.connect(lambda: self.open_file_dialog(self.onnx_bank_input))
        bank_file_layout.addWidget(self.onnx_bank_input)
        bank_file_layout.addWidget(bank_btn)
        bank_layout.addLayout(bank_file_layout, 0, 1)
        bank_group.setLayout(bank_layout)
        layout.addWidget(bank_group)

        # 检测参数
        param_group = self.create_styled_group("检测参数")
        param_grid = QGridLayout()
        param_grid.setSpacing(12)
        param_grid.addWidget(QLabel("K值:"), 0, 0)
        self.onnx_k_spin = QSpinBox()
        self.onnx_k_spin.setRange(1, 500)
        self.onnx_k_spin.setValue(25)
        param_grid.addWidget(self.onnx_k_spin, 0, 1)
        param_grid.addWidget(QLabel("图像尺寸:"), 0, 2)
        self.onnx_img_size_spin = QSpinBox()
        self.onnx_img_size_spin.setRange(128, 1024)
        self.onnx_img_size_spin.setValue(512)
        param_grid.addWidget(self.onnx_img_size_spin, 0, 3)
        param_grid.addWidget(QLabel("位置感知:"), 1, 0)
        self.onnx_pos_check = QCheckBox("启用位置感知检索")
        self.onnx_pos_check.setChecked(False)
        param_grid.addWidget(self.onnx_pos_check, 1, 1)
        param_grid.addWidget(QLabel("邻域半径:"), 1, 2)
        self.onnx_radius_spin = QSpinBox()
        self.onnx_radius_spin.setRange(0, 5)
        self.onnx_radius_spin.setValue(1)
        param_grid.addWidget(self.onnx_radius_spin, 1, 3)
        param_group.setLayout(param_grid)
        layout.addWidget(param_group)

        # 导出参数
        export_group = self.create_styled_group("导出参数")
        export_grid = QGridLayout()
        export_grid.setSpacing(12)
        export_grid.addWidget(QLabel("输出目录:"), 0, 0)
        out_layout = QHBoxLayout()
        self.onnx_output_input = QLineEdit("./onnx")
        out_btn = QPushButton("浏览")
        out_btn.setMaximumWidth(80)
        out_btn.clicked.connect(lambda: self.browse_folder(self.onnx_output_input))
        out_layout.addWidget(self.onnx_output_input)
        out_layout.addWidget(out_btn)
        export_grid.addLayout(out_layout, 0, 1)
        export_grid.addWidget(QLabel("文件名:"), 0, 2)
        self.onnx_filename_input = QLineEdit("rad_pipeline")
        self.onnx_filename_input.setPlaceholderText("不含.onnx后缀")
        export_grid.addWidget(self.onnx_filename_input, 0, 3)
        export_grid.addWidget(QLabel("Opset版本:"), 1, 0)
        self.onnx_opset_spin = QSpinBox()
        self.onnx_opset_spin.setRange(11, 20)
        self.onnx_opset_spin.setValue(14)
        export_grid.addWidget(self.onnx_opset_spin, 1, 1)
        export_grid.addWidget(QLabel("Mask尺寸:"), 2, 0)
        self.onnx_resize_spin = QSpinBox()
        self.onnx_resize_spin.setRange(64, 512)
        self.onnx_resize_spin.setValue(256)
        export_grid.addWidget(self.onnx_resize_spin, 2, 1)
        self.onnx_verify_check = QCheckBox("验证输出一致性")
        self.onnx_verify_check.setChecked(True)
        export_grid.addWidget(self.onnx_verify_check, 1, 2, 1, 2)
        export_group.setLayout(export_grid)
        layout.addWidget(export_group)

        # 设备选择
        device_group = self.create_styled_group("设备")
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("运行设备:"))
        self.onnx_device_combo = QComboBox()
        self.onnx_device_combo.addItems(['cpu', 'cuda'])
        if not torch.cuda.is_available():
            self.onnx_device_combo.setCurrentText('cpu')
        device_layout.addWidget(self.onnx_device_combo)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # 导出按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        export_btn = QPushButton("开始导出 ONNX")
        export_btn.setMinimumWidth(200)
        export_btn.setMinimumHeight(40)
        export_btn.clicked.connect(self.export_onnx)
        btn_layout.addWidget(export_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.onnx_progress = QProgressBar()
        self.onnx_progress.setMaximumHeight(6)
        layout.addWidget(self.onnx_progress)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def on_onnx_encoder_changed(self, encoder_name):
        if hasattr(self, 'onnx_weight_input') and encoder_name in _ENCODER_WEIGHT_MAP:
            self.onnx_weight_input.setText(_ENCODER_WEIGHT_MAP[encoder_name])
            self.log(f"ONNX导出: 编码器切换为 {encoder_name}，权重路径已更新")
        if hasattr(self, 'onnx_layer_input'):
            if encoder_name == 'dinov3_vitl16':
                self.onnx_layer_input.setText("5 11 17 23")
            else:
                self.onnx_layer_input.setText("3 6 9 11")

    def on_onnx_mode_changed(self, mode):
        need_bank = mode in ['rad', 'full', 'separated', 'all']
        need_encoder = mode in ['dinov3', 'full', 'separated', 'all']
        if hasattr(self, 'onnx_bank_input'):
            self.onnx_bank_input.setEnabled(need_bank)
        if hasattr(self, 'onnx_k_spin'):
            self.onnx_k_spin.setEnabled(need_bank)
        if hasattr(self, 'onnx_encoder_combo'):
            self.onnx_encoder_combo.setEnabled(need_encoder)
        if hasattr(self, 'onnx_weight_input'):
            self.onnx_weight_input.setEnabled(need_encoder)
        if hasattr(self, 'onnx_layer_input'):
            self.onnx_layer_input.setEnabled(need_encoder)
        if hasattr(self, 'onnx_img_size_spin'):
            self.onnx_img_size_spin.setEnabled(need_encoder)
        name_defaults = {
            'full': 'rad_pipeline_full',
            'dinov3': 'dinov3_encoder',
            'rad': 'rad_detector',
            'separated': 'rad_pipeline',
            'all': 'rad_pipeline',
        }
        if hasattr(self, 'onnx_filename_input'):
            self.onnx_filename_input.setText(name_defaults.get(mode, 'rad_pipeline'))

    def export_onnx(self):
        mode = self.onnx_mode_combo.currentText()
        encoder_name = self.onnx_encoder_combo.currentText()
        weight_path = self.onnx_weight_input.text()
        bank_path = self.onnx_bank_input.text()
        output_dir = self.onnx_output_input.text()

        need_encoder = mode in ['dinov3', 'full', 'separated', 'all']
        need_bank = mode in ['rad', 'full', 'separated', 'all']

        if need_encoder and not weight_path:
            QMessageBox.warning(self, "错误", "请指定编码器权重路径")
            return
        if need_encoder and not os.path.exists(weight_path):
            QMessageBox.warning(self, "错误", f"权重文件不存在:\n{weight_path}")
            return
        if need_bank and not os.path.exists(bank_path):
            QMessageBox.warning(self, "错误", f"记忆库文件不存在:\n{bank_path}")
            return
        if not output_dir:
            QMessageBox.warning(self, "错误", "请指定输出目录")
            return

        layer_str = self.onnx_layer_input.text().strip()
        try:
            layer_idx_list = [int(x) for x in layer_str.split()]
        except ValueError:
            QMessageBox.warning(self, "错误", f"层索引格式错误，请用空格分隔数字\n输入: {layer_str}")
            return

        args_dict = {
            'mode': mode,
            'encoder_name': encoder_name,
            'encoder_weight': weight_path,
            'bank_path': bank_path,
            'output_dir': output_dir,
            'output_name': self.onnx_filename_input.text().strip() or 'rad_pipeline',
            'layer_idx_list': layer_idx_list,
            'img_size': self.onnx_img_size_spin.value(),
            'k_image': self.onnx_k_spin.value(),
            'use_positional_bank': self.onnx_pos_check.isChecked(),
            'pos_radius': self.onnx_radius_spin.value(),
            'resize_mask': self.onnx_resize_spin.value(),
            'opset_version': self.onnx_opset_spin.value(),
            'verify': self.onnx_verify_check.isChecked(),
            'device': self.onnx_device_combo.currentText(),
        }

        self.log("=" * 50)
        self.log(f"开始 ONNX 导出 (模式: {mode})")
        self.onnx_progress.setValue(10)
        self.onnx_thread = ExportONNXThread(args_dict)
        self.onnx_thread.log.connect(self.log)
        self.onnx_thread.finished_signal.connect(self.on_export_onnx_finished)
        self.onnx_thread.start()
        self.log("ONNX 导出中，请稍候...")

    def on_export_onnx_finished(self, success, message):
        self.onnx_progress.setValue(100 if success else 0)
        if success:
            self.log(f"ONNX {message}")
            QMessageBox.information(self, "导出成功", message)
        else:
            self.log(f"ONNX 导出失败: {message}")
            QMessageBox.critical(self, "导出失败", f"ONNX 导出失败:\n{message}")
    
    def create_image_frame(self, title):
        """创建图像显示框架"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 2px solid #d0d0d0;
                border-radius: 6px;
            }
        """)
        frame.setMinimumSize(200, 180)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(frame)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: 11pt;
            font-weight: bold;
            color: {ModernStyle.PRIMARY_COLOR};
            background-color: transparent;
            border: none;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建图像容器
        image_container = QFrame()
        image_container.setStyleSheet("""
            background-color: white;
            border: 1px dashed #cccccc;
            border-radius: 4px;
        """)
        image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.setContentsMargins(4, 4, 4, 4)
        
        image_label = QLabel("请选择图像" if title == "原始图像" else "等待检测")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(180, 160)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_container_layout.addWidget(image_label)
        
        layout.addWidget(image_container, stretch=1)
        
        if title == "原始图像":
            self.original_label = image_label
        elif title == "异常热力图":
            self.anomaly_label = image_label
        elif title == "检测结果":
            self.overlay_label = image_label
        
        return frame
    
    def scale_pixmap_to_label(self, pixmap, label):
        """按比例缩放pixmap到label尺寸，并居中显示"""
        if pixmap.isNull():
            return pixmap
        
        margin = 16
        available_width = max(label.width() - margin, 50)
        available_height = max(label.height() - margin, 50)
        
        scaled_pixmap = pixmap.scaled(
            available_width,
            available_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        final_pixmap = QPixmap(available_width, available_height)
        final_pixmap.fill(Qt.transparent)
        
        painter = QtGui.QPainter(final_pixmap)
        x = (available_width - scaled_pixmap.width()) // 2
        y = (available_height - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()
        
        return final_pixmap
    
    def log(self, message):
        """添加日志"""
        self.log_text.append(message)
    
    def on_encoder_changed(self, encoder_name):
        """编码器选择改变时更新权重路径"""
        if hasattr(self, 'encoder_weight_map') and encoder_name in self.encoder_weight_map:
            self.weight_path_input.setText(self.encoder_weight_map[encoder_name])
            self.log(f"编码器已切换为 {encoder_name}，权重路径已自动更新")
    
    def browse_folder(self, line_edit):
        """浏览文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)
    
    def open_file_dialog(self, line_edit, file_type="权重文件", filter_str="权重文件 (*.pth *.safetensors);;All Files (*)"):
        """打开文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, f"选择{file_type}文件", "", filter_str)
        if file_path:
            line_edit.setText(file_path)
    
    def save_file_dialog(self, line_edit):
        """保存文件对话框"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "All Files (*)")
        if file_path:
            line_edit.setText(file_path)
    
    def select_test_image(self):
        """选择测试图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择测试图像", "", "Images (*.png *.jpg *.bmp)"
        )
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "错误", "文件路径不存在")
            return
        
        self.test_img_path.setText(file_path)
        self.current_image_path = file_path
        
        QTimer.singleShot(100, lambda: self.load_and_display_image(file_path))
    
    def load_and_display_image(self, file_path):
        """加载并显示图像"""
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", "图片加载失败（格式/路径错误）")
            self.original_label.setText("图片加载失败")
            return
        
        scaled_pixmap = self.scale_pixmap_to_label(pixmap, self.original_label)
        self.original_label.setPixmap(scaled_pixmap)
        
        self.anomaly_label.setText("请点击开始检测")
        self.overlay_label.setText("等待检测")
        self.single_result_text.clear()
        
        self.log(f"已选择图像: {file_path}")
        self.log(f"图像尺寸: {pixmap.width()} x {pixmap.height()}")
        self.log(f"显示尺寸: {scaled_pixmap.width()} x {scaled_pixmap.height()}")
    
    def build_memory_bank(self):
        """构建记忆库"""
        self.log("=" * 50)
        self.log("开始构建记忆库...")
        self.log("当前模式：训练样本统一缩放到 512x512，不进行裁剪")
        
        # 验证输入
        if not os.path.exists(self.data_path_input.text()):
            QMessageBox.warning(self, "错误", "数据集路径不存在")
            return
        
        categories = self.category_input.text().split()
        if not categories:
            QMessageBox.warning(self, "错误", "请输入类别名称")
            return
        
        # 验证模型名称和权重文件是否匹配
        encoder_name = self.encoder_combo.currentText()
        weight_path = self.weight_path_input.text()
        
        if encoder_name == "dinov3_vitb16" and ("vits16" in weight_path or "vitl16" in weight_path):
            QMessageBox.warning(self, "配置错误", 
                f"编码器选择为 {encoder_name} (768维)，\n"
                f"但权重文件路径包含 'vits16' (384维)或者 'vitl16' (1024维)。\n\n"
                f"请确保选择正确的权重文件，或切换编码器为 dinov3_vits16")
            return
        
        if encoder_name == "dinov3_vits16" and ("vitb16" in weight_path or "vitl16" in weight_path):
            QMessageBox.warning(self, "配置错误", 
                f"编码器选择为 {encoder_name} (384维)，\n"
                f"但权重文件路径包含 'vitb16' (768维)或者 'vitl16' (1024维)。\n\n"
                f"请确保选择正确的权重文件，或切换编码器为 dinov3_vitb16")
            return
        
        if encoder_name == "dinov3_vitl16" and ("vits16" in weight_path or "vitb16" in weight_path):
            QMessageBox.warning(self, "配置错误", 
                f"编码器选择为 {encoder_name} (1024维)，\n"
                f"但权重文件路径包含 'vits16' (384维) 或者 'vitb16' (768维)。\n\n"
                f"请确保选择正确的权重文件，或切换编码器为 dinov3_vitl16")
            return
        image_size = self.image_size_spin.value()
        
        args_dict = {
            'data_path': self.data_path_input.text(),
            'bank_path': self.bank_path_input.text(),
            'categories': categories,
            'encoder_name': self.encoder_combo.currentText(),
            'encoder_weight': self.weight_path_input.text(),
            'layers': self.layer_input.text(),
            'batch_size': self.batch_spin.value(),
            'image_size': image_size,
            'crop_size': image_size,
            'device': self.device
        }
        
        self.bank_progress.setValue(0)
        
        self.build_thread = BuildBankThread(args_dict)
        self.build_thread.log.connect(self.log)
        self.build_thread.finished_signal.connect(self.on_build_finished)
        self.build_thread.start()
        
        self.log("记忆库构建中，请稍候...")
    
    def on_build_finished(self, success, message):
        """记忆库构建完成回调"""
        self.bank_progress.setValue(100 if success else 0)
        if success:
            self.log(f"✓ {message}")
            self.log("提示：请使用新构建的 512x512 无裁剪记忆库进行评估和推理")
            QMessageBox.information(self, "完成", message)
        else:
            self.log(f"✗ 错误: {message}")
            QMessageBox.critical(self, "失败", f"记忆库构建失败:\n{message}")
    
    def run_evaluation(self):
        """运行评估"""
        self.log("=" * 50)
        self.log("开始性能评估...")
        self.log("当前模式：测试样本统一缩放到 512x512，不进行裁剪")
        
        if not os.path.exists(self.eval_bank_path.text()):
            QMessageBox.warning(self, "错误", "记忆库文件不存在")
            return
        
        # 获取当前选择的编码器和权重路径
        encoder_name = self.encoder_combo.currentText()
        encoder_weight = self.weight_path_input.text()
        
        args_dict = {
            'bank_path': self.eval_bank_path.text(),
            'data_path': self.eval_data_path.text(),
            'category': self.eval_category.text(),
            'encoder_name': encoder_name,
            'encoder_weight': encoder_weight,
            'k_image': self.k_spin.value(),
            'use_positional_bank': self.pos_check.isChecked(),
            'pos_radius': self.radius_spin.value(),
            'device': self.device
        }
        
        self.eval_progress.setValue(0)
        self.eval_thread = EvaluationThread(args_dict)
        self.eval_thread.log.connect(self.log)
        self.eval_thread.progress.connect(self.eval_progress.setValue)
        self.eval_thread.finished_signal.connect(self.on_eval_finished)
        self.eval_thread.start()
    
    def on_eval_finished(self, result):
        """评估完成回调"""
        if result.get('success'):
            result_text = f"""
评估完成！

图像级指标:
  AUROC: {result['auroc']:.4f}
  AP: {result['ap']:.4f}
  Best F1: {result['best_f1']:.4f} (阈值: {result['best_threshold']:.4f})

样本统计:
  总样本数: {result['total_samples']}
  异常样本数: {result['anomaly_samples']}
            """
            self.result_text.setText(result_text)
            self.log("✓ 评估完成")
        else:
            self.log(f"✗ 评估失败: {result.get('error', '未知错误')}")
            QMessageBox.critical(self, "失败", f"评估失败:\n{result.get('error', '未知错误')}")
    
    def run_single_test(self):
        """运行单样本测试"""
        if hasattr(self, 'detect_thread') and self.detect_thread is not None and self.detect_thread.isRunning():
            QMessageBox.warning(self, "提示", "检测正在进行中，请等待完成")
            return
        
        img_path = self.test_img_path.text()
        if not img_path:
            QMessageBox.warning(self, "错误", "请先选择测试图像")
            return
        
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "错误", "图像文件不存在")
            return
        
        if not os.path.exists(self.test_bank_path.text()):
            QMessageBox.warning(self, "错误", "记忆库文件不存在")
            return
        
        self.log("=" * 50)
        self.log(f"开始检测: {img_path}")
        self.log("当前模式：输入图像直接缩放到 512x512，不进行裁剪")
        
        self.anomaly_label.setText("正在检测...")
        self.overlay_label.setText("正在处理...")
        
        encoder_name = self.encoder_combo.currentText()
        encoder_weight = self.weight_path_input.text()
        
        args_dict = {
            'image_path': img_path,
            'bank_path': self.test_bank_path.text(),
            'encoder_name': encoder_name,
            'encoder_weight': encoder_weight,
            'k_image': self.test_k_spin.value(),
            'use_positional_bank': self.test_pos_check.isChecked(),
            'pos_radius': self.test_radius_spin.value(),
            'device': self.device,
            'fast_mode': self.fast_mode_check.isChecked(),
            'save_dir': self.save_dir_input.text().strip(),
            'threshold': self.threshold_slider.value() / 1000.0
        }

        
        self.detect_thread = DetectionThread(args_dict)
        self.detect_thread.log.connect(self.log)
        self.detect_thread.finished_signal.connect(self.on_detect_finished)
        self.detect_thread.finished_signal.connect(lambda result: self.on_detect_thread_finished(result))
        self.detect_thread.start()
    
    def on_threshold_changed(self, value):
        """阈值滑动条变化回调"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.3f}")
        
        if self.current_result and self.current_result.get('success'):
            self.update_result_with_threshold()
            self.visualize_detection_result(self.current_result, use_threshold=True, save_files=False)
    
    def update_result_with_threshold(self):
        """根据当前阈值更新结果显示和日志"""
        if not self.current_result or self.current_anomaly_map is None:
            return
        
        threshold = self.threshold_slider.value() / 100.0
        anomaly_map = self.current_anomaly_map
        
        max_score = np.max(anomaly_map)
        mean_score = np.mean(anomaly_map)
        
        above_threshold = anomaly_map > threshold
        has_anomaly = np.any(above_threshold)
        above_ratio = np.mean(above_threshold) * 100
        
        if max_score < 0.05:
            status = "正常"
        elif max_score < 0.10:
            status = "基本正常"
        elif max_score < 0.15:
            status = "疑似异常"
        else:
            status = "发现异常"
        
        warmup_time_ms = self.current_result.get('warmup_time_ms', 0.0)
        inference_time_ms = self.current_result.get('inference_time_ms', 0.0)

        result_text = f"""检测完成
最大异常分数: {max_score:.4f}
平均异常分数: {mean_score:.4f}
阈值设置: {threshold:.2f}
异常区域比例: {above_ratio:.1f}%
检测结果: {status}
Warmup时间: {warmup_time_ms:.2f} ms
单张推理时间: {inference_time_ms:.2f} ms

参考标准:
- 正常: < 0.05
- 基本正常: 0.05 - 0.10
- 疑似异常: 0.10 - 0.15
- 发现异常: > 0.15
        """
        self.single_result_text.setText(result_text)
        
        self.log(f"[检测结果] 最大分数: {max_score:.4f}, 平均分数: {mean_score:.4f}, 阈值: {threshold:.2f}, 异常区域: {above_ratio:.1f}%, 判定: {status}")
    
    def on_detect_finished(self, result):
        """检测完成回调"""
        if result.get('success'):
            self.current_result = result
            
            self.visualize_detection_result(result, use_threshold=True, save_files=False)
            self.update_result_with_threshold()
            
            self.log(
                f"✓ 检测完成，异常分数: {result['anomaly_score']:.4f} | "
                f"Warmup: {result.get('warmup_time_ms', 0):.2f} ms | "
                f"单张推理: {result.get('inference_time_ms', 0):.2f} ms"
            )
            
            orig_h, orig_w = result.get('original_size', (0, 0))
            self.log(f"[提示] 当前使用整图缩放到 512x512 进行检测，原图尺寸为 {orig_w}x{orig_h}")

            saved_paths = result.get('saved_paths', {})
            if saved_paths:
                self.log(f"保存目录: {saved_paths.get('save_dir', '')}")
                self.log(f"原图已保存: {saved_paths.get('original_path', '')}")
                self.log(f"Mask已保存: {saved_paths.get('mask_path', '')}")
                self.log(f"热力图已保存: {saved_paths.get('heatmap_path', '')}")

        else:
            error_msg = result.get('error', '未知错误')
            self.log(f"✗ 检测失败: {error_msg}")
            QMessageBox.critical(self, "失败", f"检测失败:\n{error_msg}")
            self.anomaly_label.setText("检测失败")
            self.overlay_label.setText("检测失败")
    
    def on_detect_thread_finished(self, result):
        """检测线程完成后的清理"""
        QTimer.singleShot(100, self.cleanup_detect_thread)
    
    def cleanup_detect_thread(self):
        """清理检测线程引用"""
        if hasattr(self, 'detect_thread') and self.detect_thread is not None:
            if not self.detect_thread.isRunning():
                self.detect_thread = None
    def save_display_images(self, result, anomaly_map_thresholded, cmap):
        """保存原始图像、异常热力图、检测结果图"""
        try:
            save_dir = self.save_dir_input.text().strip() if hasattr(self, 'save_dir_input') else "./inference_results"
            if not save_dir:
                save_dir = "./inference_results"
            os.makedirs(save_dir, exist_ok=True)

            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            else:
                base_name = "result"

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            original_path = os.path.join(save_dir, f"{base_name}_{timestamp}_original.png")
            heatmap_path = os.path.join(save_dir, f"{base_name}_{timestamp}_heatmap.png")
            result_path = os.path.join(save_dir, f"{base_name}_{timestamp}_result.png")

            # 1. 保存原始图像
            if self.current_image_path and os.path.exists(self.current_image_path):
                Image.open(self.current_image_path).convert("RGB").save(original_path)
            else:
                Image.fromarray(result['original_image']).save(original_path)

            # 2. 保存异常热力图（和界面中间图一致）
            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(anomaly_map_thresholded, cmap=cmap, vmin=0, vmax=0.3)
            plt.colorbar(label='异常分数', fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            plt.savefig(heatmap_path, dpi=150, pad_inches=0.02, bbox_inches='tight')
            plt.close()

            # 3. 保存检测结果图（和界面右侧图一致）
            plt.figure(figsize=(6, 6), dpi=150)
            plt.imshow(result['original_image'])
            plt.imshow(anomaly_map_thresholded, cmap=cmap, alpha=0.5, vmin=0, vmax=0.3)
            plt.axis('off')
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            plt.savefig(result_path, dpi=150, pad_inches=0.02, bbox_inches='tight')
            plt.close()

            return {
                'save_dir': save_dir,
                'original_path': original_path,
                'heatmap_path': heatmap_path,
                'result_path': result_path
            }

        except Exception as e:
            self.log(f"保存图片失败: {str(e)}")
            return {}

    def visualize_detection_result(self, result, use_threshold=True, save_files=False):
        """可视化检测结果"""
        try:
            original_img = result['original_image']
            anomaly_map = result['anomaly_map'].copy()
            
            self.current_anomaly_map = anomaly_map.copy()
            
            if use_threshold:
                threshold = self.threshold_slider.value() / 100.0
            else:
                threshold = 0.0
            
            anomaly_map_display = anomaly_map.copy()
            anomaly_map_thresholded = anomaly_map_display.copy()
            anomaly_map_thresholded[anomaly_map_thresholded < threshold] = 0
            
            display_width = max(self.anomaly_label.width() - 20, 200)
            display_height = max(self.anomaly_label.height() - 20, 200)
            
            dpi = 100
            figsize = (display_width / dpi, display_height / dpi)
            
            colors = ['blue', 'cyan', 'yellow', 'red']
            cmap = LinearSegmentedColormap.from_list('anomaly', colors, N=100)
            
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(anomaly_map_thresholded, cmap=cmap, vmin=0, vmax=0.3)
            plt.colorbar(label='异常分数', fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            heatmap_path = "./temp_heatmap.png"
            plt.savefig(heatmap_path, dpi=dpi, pad_inches=0.02)
            plt.close()
            
            heatmap_pixmap = QPixmap(heatmap_path)
            if not heatmap_pixmap.isNull():
                scaled = self.scale_pixmap_to_label(heatmap_pixmap, self.anomaly_label)
                self.anomaly_label.setPixmap(scaled)
            
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(original_img)
            plt.imshow(anomaly_map_thresholded, cmap=cmap, alpha=0.5, vmin=0, vmax=0.3)
            plt.axis('off')
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            overlay_path = "./temp_overlay.png"
            plt.savefig(overlay_path, dpi=dpi, pad_inches=0.02)
            plt.close()
            self.log(f"save_files: {save_files}")
            # 是否保存三张结果图
            if save_files is True:
                saved_paths = self.save_display_images(result, anomaly_map_thresholded, cmap)
                if saved_paths:
                    result['saved_paths'] = saved_paths
                    self.log(f"图片已保存到: {saved_paths['save_dir']}")
                    self.log(f"原始图像: {saved_paths['original_path']}")
                    self.log(f"异常热力图: {saved_paths['heatmap_path']}")
                    self.log(f"检测结果图: {saved_paths['result_path']}")

            overlay_pixmap = QPixmap(overlay_path)
            if not overlay_pixmap.isNull():
                scaled = self.scale_pixmap_to_label(overlay_pixmap, self.overlay_label)
                self.overlay_label.setPixmap(scaled)
            
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)
            if os.path.exists(overlay_path):
                os.remove(overlay_path)
                
        except Exception as e:
            self.log(f"可视化错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.anomaly_label.setText("可视化失败")
            self.overlay_label.setText("可视化失败")
    
    def resizeEvent(self, event):
        """窗口大小改变时重新缩放图片"""
        super().resizeEvent(event)
        if self.current_image_path and os.path.exists(self.current_image_path):
            pixmap = QPixmap(self.current_image_path)
            if not pixmap.isNull():
                scaled = self.scale_pixmap_to_label(pixmap, self.original_label)
                self.original_label.setPixmap(scaled)

    def save_settings(self):
        """关闭时保存所有参数到 QSettings"""
        try:
            s = self.settings
            # Tab1 — 特征记忆库 (8项)
            s.setValue("train/data_path", self.data_path_input.text())
            s.setValue("train/bank_path", self.bank_path_input.text())
            s.setValue("train/category", self.category_input.text())
            s.setValue("train/encoder", self.encoder_combo.currentText())
            s.setValue("train/weight_path", self.weight_path_input.text())
            s.setValue("train/layers", self.layer_input.text())
            s.setValue("train/batch_size", self.batch_spin.value())
            s.setValue("train/image_size", self.image_size_spin.value())
            # Tab2 — 性能评估 (6项)
            s.setValue("eval/bank_path", self.eval_bank_path.text())
            s.setValue("eval/data_path", self.eval_data_path.text())
            s.setValue("eval/category", self.eval_category.text())
            s.setValue("eval/k", self.k_spin.value())
            s.setValue("eval/radius", self.radius_spin.value())
            s.setValue("eval/positional_bank", self.pos_check.isChecked())
            # Tab3 — 单样本检测 (9项)
            s.setValue("detect/image_path", self.test_img_path.text())
            s.setValue("detect/bank_path", self.test_bank_path.text())
            s.setValue("detect/save_dir", self.save_dir_input.text())
            s.setValue("detect/k", self.test_k_spin.value())
            s.setValue("detect/radius", self.test_radius_spin.value())
            s.setValue("detect/positional_bank", self.test_pos_check.isChecked())
            s.setValue("detect/fast_mode", self.fast_mode_check.isChecked())
            s.setValue("detect/threshold", self.threshold_slider.value() / 100.0)
            # Tab4 — ONNX 导出 (15项)
            s.setValue("onnx/mode", self.onnx_mode_combo.currentText())
            s.setValue("onnx/encoder", self.onnx_encoder_combo.currentText())
            s.setValue("onnx/weight_path", self.onnx_weight_input.text())
            s.setValue("onnx/layers", self.onnx_layer_input.text())
            s.setValue("onnx/bank_path", self.onnx_bank_input.text())
            s.setValue("onnx/k", self.onnx_k_spin.value())
            s.setValue("onnx/image_size", self.onnx_img_size_spin.value())
            s.setValue("onnx/positional_bank", self.onnx_pos_check.isChecked())
            s.setValue("onnx/radius", self.onnx_radius_spin.value())
            s.setValue("onnx/output_dir", self.onnx_output_input.text())
            s.setValue("onnx/filename", self.onnx_filename_input.text())
            s.setValue("onnx/opset_version", self.onnx_opset_spin.value())
            s.setValue("onnx/resize_mask", self.onnx_resize_spin.value())
            s.setValue("onnx/verify", self.onnx_verify_check.isChecked())
            s.setValue("onnx/device", self.onnx_device_combo.currentText())
            # UI 状态 (5项)
            s.setValue("ui/last_tab", self.tabs.currentIndex())
            geom = self.normalGeometry()
            s.setValue("ui/window_pos_x", geom.x())
            s.setValue("ui/window_pos_y", geom.y())
            s.setValue("ui/window_size_w", geom.width())
            s.setValue("ui/window_size_h", geom.height())
        except Exception as e:
            print(f"Warning: failed to save settings: {e}")

    def load_settings(self):
        """启动时从 QSettings 加载参数到控件"""
        try:
            s = self.settings
            # Tab1 — 特征记忆库 (8项)
            encoder = s.value("train/encoder", "dinov3_vitb16")
            self.encoder_combo.setCurrentText(encoder)
            self.data_path_input.setText(s.value("train/data_path", "./data/mvtec"))
            self.bank_path_input.setText(s.value("train/bank_path", "./bank/memory_bank_official.pth"))
            self.category_input.setText(s.value("train/category", "metal_nut"))
            self.weight_path_input.setText(s.value("train/weight_path", _ENCODER_WEIGHT_MAP.get(encoder, "")))
            self.layer_input.setText(s.value("train/layers", "3 6 9 11"))
            self.batch_spin.setValue(int(s.value("train/batch_size", 4)))
            self.image_size_spin.setValue(int(s.value("train/image_size", 512)))
            # Trigger encoder linkage after loading encoder
            self.on_encoder_changed(encoder)
            # Tab2 — 性能评估 (6项)
            self.eval_bank_path.setText(s.value("eval/bank_path", "./bank/memory_bank_official.pth"))
            self.eval_data_path.setText(s.value("eval/data_path", "./data/mvtec"))
            self.eval_category.setText(s.value("eval/category", "metal_nut"))
            self.k_spin.setValue(int(s.value("eval/k", 30)))
            self.radius_spin.setValue(int(s.value("eval/radius", 1)))
            self.pos_check.setChecked(s.value("eval/positional_bank", True, type=bool))
            # Tab3 — 单样本检测 (9项)
            self.test_img_path.setText(s.value("detect/image_path", ""))
            self.test_bank_path.setText(s.value("detect/bank_path", "./bank/memory_bank_official.pth"))
            self.save_dir_input.setText(s.value("detect/save_dir", "./inference_results"))
            self.test_k_spin.setValue(int(s.value("detect/k", 30)))
            self.test_radius_spin.setValue(int(s.value("detect/radius", 1)))
            self.test_pos_check.setChecked(s.value("detect/positional_bank", True, type=bool))
            self.fast_mode_check.setChecked(s.value("detect/fast_mode", True, type=bool))
            threshold = float(s.value("detect/threshold", 0.10))
            self.threshold_slider.setValue(int(round(threshold * 100)))
            self.on_threshold_changed(self.threshold_slider.value())
            # Tab4 — ONNX 导出 (15项)
            onnx_mode = s.value("onnx/mode", "separated")
            self.onnx_mode_combo.setCurrentText(onnx_mode)
            onnx_encoder = s.value("onnx/encoder", "dinov3_vitb16")
            self.onnx_encoder_combo.setCurrentText(onnx_encoder)
            # Trigger linkage FIRST so defaults are set, then override with saved values
            self.on_onnx_encoder_changed(onnx_encoder)
            self.on_onnx_mode_changed(onnx_mode)
            self.onnx_weight_input.setText(s.value("onnx/weight_path", _ENCODER_WEIGHT_MAP.get(onnx_encoder, "")))
            self.onnx_layer_input.setText(s.value("onnx/layers", "3 6 9 11"))
            self.onnx_bank_input.setText(s.value("onnx/bank_path", "./bank/memory_bank_official.pth"))
            self.onnx_k_spin.setValue(int(s.value("onnx/k", 25)))
            self.onnx_img_size_spin.setValue(int(s.value("onnx/image_size", 512)))
            self.onnx_pos_check.setChecked(s.value("onnx/positional_bank", False, type=bool))
            self.onnx_radius_spin.setValue(int(s.value("onnx/radius", 1)))
            self.onnx_output_input.setText(s.value("onnx/output_dir", "./onnx"))
            self.onnx_filename_input.setText(s.value("onnx/filename", "rad_pipeline"))
            self.onnx_opset_spin.setValue(int(s.value("onnx/opset_version", 14)))
            self.onnx_resize_spin.setValue(int(s.value("onnx/resize_mask", 256)))
            self.onnx_verify_check.setChecked(s.value("onnx/verify", True, type=bool))
            self.onnx_device_combo.setCurrentText(s.value("onnx/device", "cpu"))
            # UI 状态 (5项)
            self.tabs.setCurrentIndex(int(s.value("ui/last_tab", 0)))
            pos_x = int(s.value("ui/window_pos_x", -1))
            pos_y = int(s.value("ui/window_pos_y", -1))
            size_w = int(s.value("ui/window_size_w", -1))
            size_h = int(s.value("ui/window_size_h", -1))
            if pos_x >= 0 and pos_y >= 0 and size_w > 0 and size_h > 0:
                self.setGeometry(pos_x, pos_y, size_w, size_h)
        except Exception as e:
            print(f"Warning: failed to load settings, using defaults: {e}")

    def closeEvent(self, event):
        """窗口关闭时保存所有参数"""
        self.save_settings()
        event.accept()


def main():
    """程序入口"""
    app = QApplication(sys.argv)
    
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    window = RAD_GUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
# pyinstaller --onefile --windowed --name RAD_Detector --add-data "modelscope;modelscope"   --add-data "dinov3;dinov3"  --add-data "pretrained;pretrained"   --hidden-import torch   --hidden-import torchvision  --hidden-import sklearn   --hidden-import numpy   --hidden-import PIL  --hidden-import matplotlib   --hidden-import cv2   --hidden-import safetensors   modelscope\rad_gui.py