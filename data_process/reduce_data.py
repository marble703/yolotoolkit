#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO数据集缩减工具

该脚本用于随机移除YOLO格式数据集中的一部分数据，有两种模式：
1. 按比例移除：保留指定比例的数据
2. 按数量移除：保留指定数量的数据

脚本会保持train/test/val的比例，并确保图像和标签的对应关系不被破坏。
"""

import os
import random
import shutil
import argparse
from pathlib import Path
import yaml
import glob


def count_dataset_files(dataset_path):
    """
    统计数据集中的文件数量
    """
    counts = {}
    split_dirs = ['train', 'test', 'val']
    
    for split in split_dirs:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        img_path = os.path.join(split_path, 'images')
        label_path = os.path.join(split_path, 'labels')
        
        if os.path.exists(img_path):
            img_count = len([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            img_count = 0
            
        if os.path.exists(label_path):
            label_count = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f)) and f.endswith('.txt')])
        else:
            label_count = 0
            
        counts[split] = {'images': img_count, 'labels': label_count}
    
    return counts


def get_class_distribution(dataset_path):
    """
    统计数据集中每个类别的分布情况
    """
    class_counts = {}
    split_dirs = ['train', 'test', 'val']
    
    for split in split_dirs:
        label_path = os.path.join(dataset_path, split, 'labels')
        if not os.path.exists(label_path):
            continue
            
        label_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f)) and f.endswith('.txt')]
        
        for label_file in label_files:
            try:
                with open(os.path.join(label_path, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id not in class_counts:
                                class_counts[class_id] = {'train': 0, 'test': 0, 'val': 0, 'total': 0}
                            class_counts[class_id][split] += 1
                            class_counts[class_id]['total'] += 1
            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
    
    return class_counts


def reduce_by_ratio(dataset_path, output_path, keep_ratio=0.5):
    """
    按比例减少数据集
    
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出数据集路径
        keep_ratio: 保留的数据比例 (0.0 到 1.0)
    """
    print(f"按比例减少数据集：保留 {keep_ratio * 100:.1f}% 的数据")
    
    if keep_ratio <= 0 or keep_ratio >= 1:
        print("保留比例必须在 0 到 1 之间")
        return False
    
    # 复制data.yaml文件
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_path):
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(output_path, 'data.yaml'))
    
    # 遍历train, test, val目录
    split_dirs = ['train', 'test', 'val']
    for split in split_dirs:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
        
        # 创建输出目录
        out_split_path = os.path.join(output_path, split)
        out_img_path = os.path.join(out_split_path, 'images')
        out_label_path = os.path.join(out_split_path, 'labels')
        os.makedirs(out_img_path, exist_ok=True)
        os.makedirs(out_label_path, exist_ok=True)
        
        # 获取图像文件列表
        img_path = os.path.join(split_path, 'images')
        if not os.path.exists(img_path):
            continue
            
        img_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and 
                     f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 随机选择要保留的文件
        num_keep = max(1, int(len(img_files) * keep_ratio))
        keep_files = random.sample(img_files, num_keep)
        
        print(f"{split}集: 从 {len(img_files)} 减至 {num_keep} 个样本")
        
        # 复制选中的文件
        for img_file in keep_files:
            # 复制图像
            shutil.copy2(os.path.join(img_path, img_file), os.path.join(out_img_path, img_file))
            
            # 复制对应的标签文件
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(split_path, 'labels', label_file)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(out_label_path, label_file))
    
    print(f"处理完成，缩减后的数据集已保存到: {output_path}")
    return True


def reduce_to_count(dataset_path, output_path, target_count=100, per_class=False):
    """
    减少数据集至指定数量
    
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出数据集路径
        target_count: 目标文件数量
        per_class: 是否按每个类别计算数量
    """
    split_dirs = ['train', 'test', 'val']
    if per_class:
        print(f"按类别减少数据集：每个类别保留 {target_count} 个样本")
        # 获取类别分布
        class_counts = get_class_distribution(dataset_path)
        if not class_counts:
            print("无法获取类别分布信息")
            return False
        
        # 显示类别分布
        print("\n类别分布情况:")
        for class_id, counts in sorted(class_counts.items()):
            print(f"类别 {class_id}: 总共 {counts['total']} 个样本 (train: {counts['train']}, val: {counts['val']}, test: {counts['test']})")
        
        # 为每个类别创建跟踪计数器
        class_counters = {class_id: 0 for class_id in class_counts.keys()}
        
        # 跟踪每个文件包含的类别
        file_classes = {}
        
        for split in split_dirs:
            label_path = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(label_path):
                continue
                
            label_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f)) and f.endswith('.txt')]
            
            for label_file in label_files:
                file_key = (split, label_file)
                file_classes[file_key] = set()
                
                try:
                    with open(os.path.join(label_path, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                file_classes[file_key].add(class_id)
                except Exception as e:
                    print(f"处理文件 {label_file} 时出错: {e}")
    else:
        print(f"减少数据集至固定数量：保留总共 {target_count} 个样本")
    
    # 复制data.yaml文件
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(yaml_path):
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(output_path, 'data.yaml'))
    
    # 处理前统计
    original_counts = count_dataset_files(dataset_path)
    total_original = sum(counts['images'] for counts in original_counts.values())
    
    # 计算每个分割集应该保留的样本数
    split_ratios = {}
    for split, counts in original_counts.items():
        if total_original > 0:
            split_ratios[split] = counts['images'] / total_original
        else:
            split_ratios[split] = 0
    
    # 遍历train, test, val目录
    keep_files_by_split = {}
    
    if per_class:
        # 按类别选择文件
        for class_id in class_counts.keys():
            for split in split_dirs:
                split_path = os.path.join(dataset_path, split)
                if not os.path.exists(split_path):
                    continue
                
                # 找出包含此类别的文件
                class_files = []
                for file_key, classes in file_classes.items():
                    if file_key[0] == split and class_id in classes:
                        class_files.append(os.path.splitext(file_key[1])[0])
                
                # 计算这个分割集中这个类别应该保留多少个
                split_class_count = class_counts[class_id][split]
                if split_class_count > 0:
                    class_split_ratio = split_class_count / class_counts[class_id]['total']
                    num_keep = min(split_class_count, int(target_count * class_split_ratio))
                    
                    if num_keep > 0 and class_files:
                        # 随机选择要保留的文件
                        selected_files = random.sample(class_files, min(num_keep, len(class_files)))
                        
                        if split not in keep_files_by_split:
                            keep_files_by_split[split] = set()
                        keep_files_by_split[split].update(selected_files)
    else:
        # 按总数选择文件
        for split in split_dirs:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
            
            # 获取图像文件列表
            img_path = os.path.join(split_path, 'images')
            if not os.path.exists(img_path):
                continue
                
            img_files = [os.path.splitext(f)[0] for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and 
                         f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # 计算这个分割集应该保留多少个
            num_keep = max(1, int(target_count * split_ratios[split]))
            num_keep = min(num_keep, len(img_files))
            
            # 随机选择要保留的文件
            keep_files_by_split[split] = set(random.sample(img_files, num_keep))
    
    # 复制选中的文件
    for split, base_names in keep_files_by_split.items():
        # 创建输出目录
        out_split_path = os.path.join(output_path, split)
        out_img_path = os.path.join(out_split_path, 'images')
        out_label_path = os.path.join(out_split_path, 'labels')
        os.makedirs(out_img_path, exist_ok=True)
        os.makedirs(out_label_path, exist_ok=True)
        
        print(f"{split}集: 保留 {len(base_names)} 个样本")
        
        # 图像目录路径
        img_path = os.path.join(dataset_path, split, 'images')
        
        # 复制文件
        for base_name in base_names:
            # 查找匹配的图像文件
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            img_file = None
            for ext in img_extensions:
                potential_file = f"{base_name}{ext}"
                if os.path.exists(os.path.join(img_path, potential_file)):
                    img_file = potential_file
                    break
            
            if img_file:
                # 复制图像
                shutil.copy2(os.path.join(img_path, img_file), os.path.join(out_img_path, img_file))
                
                # 复制对应的标签文件
                label_file = f"{base_name}.txt"
                label_path = os.path.join(dataset_path, split, 'labels', label_file)
                if os.path.exists(label_path):
                    shutil.copy2(label_path, os.path.join(out_label_path, label_file))
    
    print(f"处理完成，缩减后的数据集已保存到: {output_path}")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO数据集缩减工具')
    parser.add_argument('--dataset', type=str, required=True, help='原始数据集路径')
    parser.add_argument('--output', type=str, required=True, help='输出数据集路径')
    
    subparsers = parser.add_subparsers(dest='mode', help='减少模式')
    
    # 按比例减少
    ratio_parser = subparsers.add_parser('ratio', help='按比例减少')
    ratio_parser.add_argument('--keep', type=float, required=True, help='保留的数据比例 (0.0 到 1.0)')
    
    # 减少到指定数量
    count_parser = subparsers.add_parser('count', help='减少到指定数量')
    count_parser.add_argument('--num', type=int, required=True, help='保留的样本数量')
    count_parser.add_argument('--per-class', action='store_true', help='每个类别保留指定数量')
    
    return parser.parse_args()


def main():
    # =============================================
    # 可调整参数 - 直接在代码中修改这些值
    # =============================================
    # 数据集路径 
    dataset_path = "external_datasets/Ball_set.v4i.yolov8_no_tennis/splited"  # 原始数据集路径
    output_path = "external_datasets/Ball_set.v4i.yolov8_no_tennis_reduced"  # 输出数据集路径
    
    # 减少模式: 'ratio' - 按比例减少, 'count' - 按数量减少
    mode = 'count'
    
    # 按比例减少模式的参数
    keep_ratio = 0.3  # 保留的数据比例 (0.0 到 1.0)
    
    # 按数量减少模式的参数
    target_count = 600  # 保留的样本数量
    per_class = False   # 是否按每个类别计算数量
    # =============================================
    
    # 是否使用命令行参数 (设为False则使用上面设置的参数)
    use_cli_args = False
    
    if use_cli_args:
        args = parse_arguments()
        dataset_path = args.dataset
        output_path = args.output
        mode = args.mode
        
        if mode == 'ratio':
            keep_ratio = args.keep
        elif mode == 'count':
            target_count = args.num
            per_class = args.per_class
    
    # 验证路径
    if not os.path.exists(dataset_path):
        print(f"错误：数据集路径 {dataset_path} 不存在")
        return
    
    if os.path.exists(output_path):
        print(f"警告：输出路径 {output_path} 已存在，可能会覆盖现有文件")
        confirm = input("是否继续？ (y/n): ")
        if confirm.lower() != 'y':
            return
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 根据模式执行相应的操作
    if mode == 'ratio':
        reduce_by_ratio(dataset_path, output_path, keep_ratio)
    elif mode == 'count':
        reduce_to_count(dataset_path, output_path, target_count, per_class)
    else:
        print("请指定减少模式：ratio 或 count")


if __name__ == '__main__':
    main()