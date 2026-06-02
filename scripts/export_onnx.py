"""
RAD + DINOv3 ONNX 导出脚本 CLI 入口。

核心导出逻辑在 modelscope/onnx_export.py 中。
"""

import os
import sys
import argparse

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

from modelscope.onnx_export import (
    load_encoder, export_dinov3_onnx, export_rad_onnx,
    export_full_pipeline_onnx,
)
from modelscope.config import (
    LAYER_CONFIGS, DEFAULT_ONNX_IMG_SIZE, DEFAULT_RESIZE_MASK,
    DEFAULT_BANK_PATH, DEFAULT_ONNX_DIR, find_weight_path,
)


def main():
    parser = argparse.ArgumentParser(description='RAD + DINOv3 ONNX 导出工具（支持 .pth 和 .safetensors 格式）')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['dinov3', 'rad', 'full', 'separated', 'all'],
                        help='导出模式: dinov3=仅编码器, rad=仅检测器, full=完整pipeline, separated=编码器+检测器(推荐), all=全部')
    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16',
                        help='编码器名称: dinov3_vitb16, dinov3_vits16, dinov3_vitl16')
    parser.add_argument('--encoder_weight', type=str, default=None,
                        help='编码器权重路径 (.pth 或 .safetensors)，不指定则自动搜索')
    parser.add_argument('--bank_path', type=str, default=DEFAULT_BANK_PATH,
                        help='记忆库文件路径')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_ONNX_DIR,
                        help='ONNX 模型输出目录')
    parser.add_argument('--layer_idx_list', nargs='+', type=int, default=[3, 6, 9, 11],
                        help='层索引列表 (vitb16/vits16默认[3,6,9,11], vitl16默认[5,11,17,23])')
    parser.add_argument('--img_size', type=int, default=DEFAULT_ONNX_IMG_SIZE,
                        help='输入图像尺寸')
    parser.add_argument('--k_image', type=int, default=25,
                        help='图像级最近邻数量')
    parser.add_argument('--use_positional_bank', action='store_true', default=False,
                        help='是否使用位置感知')
    parser.add_argument('--pos_radius', type=int, default=3,
                        help='位置感知半径')
    parser.add_argument('--resize_mask', type=int, default=DEFAULT_RESIZE_MASK,
                        help='异常图上采样尺寸')
    parser.add_argument('--layer_weights', nargs='+', type=float, default=None,
                        help='层权重')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset 版本')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='是否验证输出一致性')
    parser.add_argument('--verify_pytorch', action='store_true', default=True,
                        help='是否与原始 PyTorch RADDetector 对比验证')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备')
    parser.add_argument('--low_memory', action='store_true', default=False,
                        help='低内存模式（禁用完整pipeline验证）')

    args = parser.parse_args()

    # 自动调整层索引
    if args.layer_idx_list == [3, 6, 9, 11] and args.encoder_name == 'dinov3_vitl16':
        args.layer_idx_list = LAYER_CONFIGS['dinov3_vitl16']
        print(f"ViT-L16 自动调整层索引为: {args.layer_idx_list}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("RAD + DINOv3 ONNX 导出工具（支持 .pth 和 .safetensors）")
    print("=" * 70)

    encoder = None
    if args.mode in ['dinov3', 'all', 'full', 'separated']:
        print(f"\n加载编码器: {args.encoder_name}")

        # 确定权重路径
        if args.encoder_weight is None:
            try:
                args.encoder_weight = find_weight_path(args.encoder_name)
            except FileNotFoundError:
                print(f"错误: 未找到 {args.encoder_name} 的权重文件")
                return

        print(f"权重文件: {args.encoder_weight}")
        encoder = load_encoder(args.encoder_name, args.encoder_weight, device=args.device)

        # 显示模型信息
        print(f"  embed_dim: {encoder.embed_dim}")
        print(f"  n_storage_tokens: {encoder.n_storage_tokens}")
        is_ms = hasattr(encoder, 'layer') and not hasattr(encoder, 'blocks')
        print(f"  架构: {'ModelScope' if is_ms else 'Official'}")

    success = True

    if args.mode == 'dinov3':
        success = export_dinov3_onnx(
            encoder=encoder,
            output_path=os.path.join(args.output_dir, 'dinov3_encoder.onnx'),
            layer_indices=args.layer_idx_list,
            img_size=args.img_size,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

    elif args.mode == 'rad':
        success = export_rad_onnx(
            memory_bank_path=args.bank_path,
            output_path=os.path.join(args.output_dir, 'rad_detector.onnx'),
            k_image=args.k_image,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            resize_mask=args.resize_mask,
            layer_weights=args.layer_weights,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

    elif args.mode == 'separated':
        print("\n" + "=" * 70)
        print("导出 DINOv3 编码器")
        print("=" * 70)
        success &= export_dinov3_onnx(
            encoder=encoder,
            output_path=os.path.join(args.output_dir, 'dinov3_encoder.onnx'),
            layer_indices=args.layer_idx_list,
            img_size=args.img_size,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

        print("\n" + "=" * 70)
        print("导出 RAD 检测器")
        print("=" * 70)
        success &= export_rad_onnx(
            memory_bank_path=args.bank_path,
            output_path=os.path.join(args.output_dir, 'rad_detector.onnx'),
            k_image=args.k_image,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            resize_mask=args.resize_mask,
            layer_weights=args.layer_weights,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

    elif args.mode == 'full':
        success = export_full_pipeline_onnx(
            encoder=encoder,
            memory_bank_path=args.bank_path,
            output_path=os.path.join(args.output_dir, 'rad_pipeline_full_selected_3579.onnx'),
            layer_indices=args.layer_idx_list,
            img_size=args.img_size,
            k_image=args.k_image,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            resize_mask=args.resize_mask,
            layer_weights=args.layer_weights,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
            low_memory=args.low_memory,
        )

    elif args.mode == 'all':
        print("\n" + "=" * 70)
        print("导出 DINOv3 编码器")
        print("=" * 70)
        success &= export_dinov3_onnx(
            encoder=encoder,
            output_path=os.path.join(args.output_dir, 'dinov3_encoder.onnx'),
            layer_indices=args.layer_idx_list,
            img_size=args.img_size,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

        print("\n" + "=" * 70)
        print("导出 RAD 检测器")
        print("=" * 70)
        success &= export_rad_onnx(
            memory_bank_path=args.bank_path,
            output_path=os.path.join(args.output_dir, 'rad_detector.onnx'),
            k_image=args.k_image,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            resize_mask=args.resize_mask,
            layer_weights=args.layer_weights,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
        )

        print("\n" + "=" * 70)
        print("导出完整 Pipeline（可选，需要较多内存）")
        print("=" * 70)
        success &= export_full_pipeline_onnx(
            encoder=encoder,
            memory_bank_path=args.bank_path,
            output_path=os.path.join(args.output_dir, 'rad_pipeline_full.onnx'),
            layer_indices=args.layer_idx_list,
            img_size=args.img_size,
            k_image=args.k_image,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            resize_mask=args.resize_mask,
            layer_weights=args.layer_weights,
            opset_version=args.opset_version,
            verify=args.verify,
            device=args.device,
            low_memory=args.low_memory,
        )

    print("\n" + "=" * 70)
    if success:
        print("导出成功！")
        print(f"ONNX 模型保存在: {args.output_dir}")
        if args.mode in ['separated', 'all']:
            print("\n推荐使用方式:")
            print("  python onnx_inference.py --mode separated")
    else:
        print("导出失败，请检查错误信息")
    print("=" * 70)


if __name__ == '__main__':
    main()
