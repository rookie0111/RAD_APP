"""
RAD ONNX 推理脚本 (Full Pipeline 模式)

支持单张图片测试和文件夹批量处理，输出:
  - 最大异常分数和 OK/NG 判定
  - 原始图像、异常热力图、检测结果叠加图 (与GUI一致)
  - 推理速度统计

支持 CPU 和 GPU 推理 (通过 --device 参数选择)

用法:
  # 单张图片 (CPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg

  # 单张图片 (GPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg --device cuda

  # 文件夹批量 (GPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image_dir ./data/mvtec/fupei/test/ng --device cuda

  # 调整参数
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg --threshold 0.12 --save_vis
"""

import os
import sys

_project_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_project_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from modelscope._path_setup import setup_sys_path
setup_sys_path()

# Suppress ONNX Runtime logging (must be set before importing onnxruntime)
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

import time
import argparse
import numpy as np
from PIL import Image

from modelscope.config import IMAGENET_MEAN_NP, IMAGENET_STD_NP, DEFAULT_IMAGE_SIZE, DEFAULT_THRESHOLD


def preprocess_image(image_path, img_size=DEFAULT_IMAGE_SIZE):
    """预处理图像: resize -> to_tensor -> normalize"""
    mean = IMAGENET_MEAN_NP
    std = IMAGENET_STD_NP

    img = Image.open(image_path).convert('RGB')
    original_w, original_h = img.size
    img = img.resize((img_size, img_size), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    return arr, (original_h, original_w)


def postprocess(anomaly_map, original_size, threshold):
    """后处理: 上采样到原图尺寸，生成热力图和叠加图"""
    original_h, original_w = original_size

    anomaly_map_resized = np.resize(anomaly_map, (original_h, original_w))

    max_score = float(np.max(anomaly_map_resized))
    mean_score = float(np.mean(anomaly_map_resized))
    above_threshold = anomaly_map_resized > threshold
    anomaly_ratio = float(np.mean(above_threshold)) * 100

    if max_score < 0.05:
        status = "OK"
    elif max_score < threshold * 0.5:
        status = "OK"
    elif max_score < threshold:
        status = "REVIEW"
    else:
        status = "NG"

    return {
        'max_score': max_score,
        'mean_score': mean_score,
        'anomaly_ratio': anomaly_ratio,
        'status': status,
        'anomaly_map': anomaly_map_resized,
    }


def save_visualization(image_path, result, output_dir, threshold=0.10):
    """保存可视化结果: 原图、热力图、叠加图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    original_img = Image.open(image_path).convert('RGB')
    original_arr = np.array(original_img)

    anomaly_map = result['anomaly_map']
    anomaly_map_vis = anomaly_map.copy()
    anomaly_map_vis[anomaly_map_vis < threshold] = 0

    colors = ['blue', 'cyan', 'yellow', 'red']
    cmap = LinearSegmentedColormap.from_list('anomaly', colors, N=100)

    # 1. 原始图像
    original_save_path = os.path.join(output_dir, f"{base_name}_original.png")
    original_img.save(original_save_path)

    # 2. 异常热力图
    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(anomaly_map_vis, cmap=cmap, vmin=0, vmax=0.3)
    cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score')
    ax.set_title(f"Max: {result['max_score']:.4f} | {result['status']}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # 3. 检测结果叠加图
    overlay_path = os.path.join(output_dir, f"{base_name}_result.png")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(original_arr)
    ax.imshow(anomaly_map_vis, cmap=cmap, alpha=0.5, vmin=0, vmax=0.3)
    ax.set_title(f"Max: {result['max_score']:.4f} | {result['status']}", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    return {
        'original': original_save_path,
        'heatmap': heatmap_path,
        'overlay': overlay_path,
    }


def create_session(onnx_path, device='cpu'):
    """创建 ONNX Runtime 会话，支持 CPU 和 GPU
    
    GPU推理失败时自动回退到CPU。
    """
    import onnxruntime as ort

    if device == 'cuda':
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            try:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    }),
                    'CPUExecutionProvider',
                ]
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=providers,
                )

                if 'CUDAExecutionProvider' in session.get_providers():
                    print(f"  使用 GPU 推理 (CUDA)")
                    return session, True
                else:
                    print(f"  GPU 推理未激活, 回退到 CPU")
                    del session
            except Exception as e:
                print(f"  GPU 会话创建失败: {e}")
                print(f"  回退到 CPU 推理")

        print(f"  CUDA 不可用, 使用 CPU 推理")

    # CPU fallback
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    print(f"  使用 CPU 推理")
    return session, False


def run_inference(onnx_path, image_path, threshold=0.10, img_size=512,
                  save_vis=False, output_dir=None, warmup=3, device='cpu'):
    """单张图片推理"""
    session, use_cuda = create_session(onnx_path, device)
    input_name = session.get_inputs()[0].name

    # Warmup
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # 预处理
    input_tensor, original_size = preprocess_image(image_path, img_size)

    # 推理 (GPU失败自动回退CPU)
    dev_label = "GPU" if use_cuda else "CPU"
    try:
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000.0
    except RuntimeError as e:
        if use_cuda and ('CUDA' in str(e) or 'CUDAExecutionProvider' in str(e) or 'cuDNN' in str(e) or 'CUDNN' in str(e)):
            print(f"  GPU推理失败, 回退到CPU...")
            del session
            session, use_cuda = create_session(onnx_path, 'cpu')
            input_name = session.get_inputs()[0].name
            dev_label = "CPU"
            for _ in range(warmup):
                session.run(None, {input_name: dummy})
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = (time.perf_counter() - start_time) * 1000.0
        else:
            raise

    anomaly_map = outputs[0][0]

    # 后处理
    result = postprocess(anomaly_map, original_size, threshold)
    result['inference_time_ms'] = inference_time

    print(f"  图片: {os.path.basename(image_path)}")
    print(f"  最大异常分数: {result['max_score']:.4f}")
    print(f"  平均异常分数: {result['mean_score']:.4f}")
    print(f"  异常区域比例: {result['anomaly_ratio']:.1f}%")
    print(f"  阈值: {threshold:.2f}")
    print(f"  判定: {result['status']}")
    print(f"  推理时间: {inference_time:.2f} ms ({dev_label})")

    if save_vis and output_dir:
        paths = save_visualization(image_path, result, output_dir, threshold)
        print(f"  已保存可视化结果:")
        print(f"    原图: {paths['original']}")
        print(f"    热力图: {paths['heatmap']}")
        print(f"    叠加图: {paths['overlay']}")

    return result


def run_batch_inference(onnx_path, image_dir, threshold=0.10, img_size=512,
                        save_vis=False, output_dir=None, warmup=3, device='cpu'):
    """文件夹批量推理"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for f in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(image_dir, f))

    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图片文件")
        return []

    print(f"找到 {len(image_files)} 张图片")
    print(f"ONNX模型: {onnx_path}")
    print(f"设备: {device.upper()}")
    print(f"阈值: {threshold}")
    print("=" * 70)

    session, use_cuda = create_session(onnx_path, device)
    input_name = session.get_inputs()[0].name
    dev_label = "GPU" if use_cuda else "CPU"

    # Warmup
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    # GPU推理失败时回退用的CPU session
    cpu_session = None

    results = []
    total_time = 0.0
    ok_count = 0
    ng_count = 0

    for i, image_path in enumerate(image_files):
        input_tensor, original_size = preprocess_image(image_path, img_size)

        try:
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = (time.perf_counter() - start_time) * 1000.0
        except RuntimeError as e:
            if use_cuda and ('CUDA' in str(e) or 'cuDNN' in str(e) or 'CUDNN' in str(e)):
                print(f"  GPU推理失败, 回退到CPU...")
                if cpu_session is None:
                    cpu_session, _ = create_session(onnx_path, 'cpu')
                    cpu_input_name = cpu_session.get_inputs()[0].name
                    for _ in range(warmup):
                        cpu_session.run(None, {cpu_input_name: dummy})
                session = cpu_session
                input_name = cpu_input_name
                use_cuda = False
                dev_label = "CPU"
                start_time = time.perf_counter()
                outputs = session.run(None, {input_name: input_tensor})
                inference_time = (time.perf_counter() - start_time) * 1000.0
            else:
                raise

        anomaly_map = outputs[0][0]
        total_time += inference_time

        result = postprocess(anomaly_map, original_size, threshold)
        result['image_path'] = image_path
        result['inference_time_ms'] = inference_time
        results.append(result)

        if result['status'] == 'OK':
            ok_count += 1
        else:
            ng_count += 1

        short_name = os.path.basename(image_path)
        status_str = f"{'OK':>4}" if result['status'] == 'OK' else result['status']
        print(f"  [{i+1:3d}/{len(image_files)}] {short_name:<30s} "
              f"score={result['max_score']:.4f}  {status_str}  "
              f"{inference_time:.1f}ms ({dev_label})")

        if save_vis and output_dir:
            save_visualization(image_path, result, output_dir, threshold)

    avg_time = total_time / len(image_files)
    print("=" * 70)
    print(f"总计: {len(image_files)} 张图片")
    print(f"  OK: {ok_count}, NG: {ng_count}")
    print(f"  设备: {dev_label}")
    print(f"  平均推理时间: {avg_time:.2f} ms/张")
    print(f"  总耗时: {total_time:.1f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='RAD ONNX Full Pipeline 推理 (支持单张和批量, CPU/GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片推理 (CPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg

  # 单张图片推理 (GPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg --device cuda

  # 文件夹批量推理 (GPU)
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image_dir ./data/test/ng --device cuda

  # 保存可视化结果
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image test.jpg --save_vis --output_dir ./results

  # 调整阈值
  python onnx_inference_full.py --onnx_path onnx/rad_pipeline_full.onnx --image_dir ./data/test/ng --threshold 0.12 --device cuda
        """)

    parser.add_argument('--onnx_path', type=str, required=True,
                        help='ONNX模型路径 (rad_pipeline_full.onnx)')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='图片文件夹路径 (批量模式)')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='OK/NG判定阈值 (default: 0.10)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='输入图像尺寸 (default: 512)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='推理设备: cpu 或 cuda (default: cpu)')
    parser.add_argument('--save_vis', action='store_true', default=False,
                        help='保存可视化结果 (原图+热力图+叠加图)')
    parser.add_argument('--output_dir', type=str, default='./onnx_results',
                        help='可视化结果输出目录')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup次数 (default: 3)')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error('请指定 --image 或 --image_dir')

    if not os.path.exists(args.onnx_path):
        print(f"错误: ONNX模型不存在: {args.onnx_path}")
        return

    print("=" * 70)
    print("RAD ONNX Full Pipeline 推理")
    print("=" * 70)
    print(f"设备: {args.device.upper()}")

    if args.image:
        print(f"\n模式: 单张图片推理")
        print(f"图片: {args.image}")
        run_inference(
            onnx_path=args.onnx_path,
            image_path=args.image,
            threshold=args.threshold,
            img_size=args.img_size,
            save_vis=args.save_vis,
            output_dir=args.output_dir,
            warmup=args.warmup,
            device=args.device,
        )
    elif args.image_dir:
        print(f"\n模式: 批量推理")
        print(f"目录: {args.image_dir}")
        run_batch_inference(
            onnx_path=args.onnx_path,
            image_dir=args.image_dir,
            threshold=args.threshold,
            img_size=args.img_size,
            save_vis=args.save_vis,
            output_dir=args.output_dir,
            warmup=args.warmup,
            device=args.device,
        )


if __name__ == '__main__':
    main()