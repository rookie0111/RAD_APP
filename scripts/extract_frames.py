"""
视频抽帧工具：从视频中均匀抽取 N 帧，按 YOLO 格式划分训练/验证集。

用法:
  python data_clening/extract_frames.py --video "C:\path\to\video.mp4" --n_frames 200 --out_dir "C:\path\to\yolo_dataset"
"""

import os
import sys
import argparse
import random
import shutil
import cv2


def extract_frames(video_path, n_frames=200):
    """从视频中均匀抽取 n_frames 帧，返回帧列表 [(timestamp, frame)]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"视频总帧数: {total_frames}, FPS: {fps:.2f}, 时长: {duration:.1f}s")

    if total_frames <= n_frames:
        # 如果视频帧数不足，全部抽取
        indices = list(range(total_frames))
        print(f"视频帧数({total_frames})不足，抽取全部帧")
    else:
        # 均匀抽取 n_frames 帧
        step = total_frames / n_frames
        indices = [int(i * step) for i in range(n_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps if fps > 0 else idx
            frames.append((timestamp, frame))
        else:
            print(f"  警告: 无法读取第 {idx} 帧")

    cap.release()
    print(f"成功抽取 {len(frames)} 帧")
    return frames


def main():
    parser = argparse.ArgumentParser(description='视频抽帧并划分 YOLO 数据集')
    parser.add_argument('--video', required=True, help='视频文件路径')
    parser.add_argument('--n_frames', type=int, default=200, help='抽取帧数（默认 200）')
    parser.add_argument('--out_dir', required=True, help='输出目录（YOLO 格式根目录）')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例（默认 0.2）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        sys.exit(1)

    # 抽帧
    frames = extract_frames(args.video, args.n_frames)

    # 随机打乱
    random.seed(args.seed)
    random.shuffle(frames)

    # 按比例划分
    n_val = max(1, int(len(frames) * args.val_ratio))
    n_train = len(frames) - n_val
    val_frames = frames[:n_val]
    train_frames = frames[n_val:]
    print(f"划分: 训练集 {n_train} 张, 验证集 {n_val} 张")

    # 创建 YOLO 目录结构
    out_dir = args.out_dir
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # 保存训练集
    for i, (ts, frame) in enumerate(train_frames):
        fname = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, 'images/train', fname), frame)
        # 创建空标签文件（后续手动标注）
        with open(os.path.join(out_dir, 'labels/train', fname.replace('.jpg', '.txt')), 'w') as f:
            pass

    # 保存验证集
    for i, (ts, frame) in enumerate(val_frames):
        fname = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, 'images/val', fname), frame)
        # 创建空标签文件
        with open(os.path.join(out_dir, 'labels/val', fname.replace('.jpg', '.txt')), 'w') as f:
            pass

    print(f"\n数据集已生成: {out_dir}")
    print(f"  训练集: images/train/ ({n_train} 张)")
    print(f"  验证集: images/val/ ({n_val} 张)")
    print(f"  标签文件: labels/train/, labels/val/")
    print(f"\n提示: 标签文件为空的 .txt 文件，请使用 LabelImg/LabelStudio 等工具标注后填充。")


if __name__ == '__main__':
    main()
