#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将YOLO格式数据集转换为COCO格式，并支持创建新图像文件夹和重命名图像
"""

import os
import json
import yaml
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil


def create_image_info(image_id, file_name, image_size):
    """创建图像信息字典"""
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "license": 1
    }
    return image_info


def create_annotation(annotation_id, image_id, category_id, bbox, segmentation=None):
    """创建标注信息字典"""
    x, y, w, h = bbox
    area = w * h
    
    # COCO格式的框是[x,y,width,height]
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": area,
        "segmentation": segmentation if segmentation else [],
        "iscrowd": 0
    }
    return annotation


def convert_yolo_to_coco(yolo_dir, output_file, class_file=None, create_new_images=False, new_images_dir=None):
    """
    将YOLO格式数据集转换为COCO格式，可选择创建新图像文件夹并处理重名问题
    
    参数:
        yolo_dir: YOLO数据集目录，包含images和labels子目录
        output_file: 输出的COCO格式JSON文件路径
        class_file: 类别文件路径，如果为None，则尝试从data.yaml读取
        create_new_images: 是否创建新图像文件夹
        new_images_dir: 新图像文件夹路径，如果为None，则使用output_file的目录 + "_images"
    """
    # 初始化COCO数据集字典
    coco_output = {
        "info": {
            "description": "Converted from YOLO format",
            "contributor": "Converter",
            "year": 2025
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 创建目录路径
    yolo_path = Path(yolo_dir)
    
    # 检查是否存在子集目录（train/test/val）
    has_subsets = False
    subset_dirs = ["train", "test", "val"]
    valid_subsets = []
    
    for subset in subset_dirs:
        if (yolo_path / subset).exists():
            valid_subsets.append(subset)
            has_subsets = True
    
    if not valid_subsets and not has_subsets:
        # 没有子集目录，查找images目录
        if (yolo_path / "images").exists():
            image_dirs = {"dataset": yolo_path / "images"}
            label_dirs = {"dataset": yolo_path / "labels"}
        else:
            raise FileNotFoundError(f"无法找到images目录或train/test/val子目录，请检查路径: {yolo_dir}")
    else:
        # 有子集目录，为每个子集创建路径
        image_dirs = {}
        label_dirs = {}
        
        for subset in valid_subsets:
            subset_path = yolo_path / subset
            
            # 检查子集下是否有images目录
            if (subset_path / "images").exists():
                image_dirs[subset] = subset_path / "images"
                label_dirs[subset] = subset_path / "labels"
            else:
                # 如果子集下没有images目录，则直接使用子集目录
                image_dirs[subset] = subset_path
                
                # 查找标签目录
                if (subset_path / "labels").exists():
                    label_dirs[subset] = subset_path / "labels"
                elif (yolo_path / "labels").exists():
                    # 有时标签在顶层目录
                    label_dirs[subset] = yolo_path / "labels"
                else:
                    raise FileNotFoundError(f"无法找到{subset}子集对应的labels目录")
    
    # 读取类别
    classes = []
    if class_file:
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        # 尝试从data.yaml读取
        yaml_file = yolo_path / "data.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        classes = data['names']
            except Exception as e:
                print(f"读取data.yaml时出错: {e}")
        
        # 如果仍未找到类别，尝试在上一级目录找data.yaml
        if not classes:
            yaml_file = yolo_path.parent / "data.yaml"
            if yaml_file.exists():
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            classes = data['names']
                except Exception as e:
                    print(f"读取上级目录data.yaml时出错: {e}")
    
    # 如果仍未找到类别，抛出错误
    if not classes:
        raise ValueError("未能读取类别信息，请提供classes.txt文件或确保data.yaml包含类别信息")
    
    # 设置类别
    for i, cls in enumerate(classes):
        category = {
            "id": i,
            "name": cls,
            "supercategory": "none"
        }
        coco_output["categories"].append(category)
    
    # 创建新图像目录（如果需要）
    if create_new_images:
        if new_images_dir is None:
            # 如果未指定新图像目录，使用输出文件的目录 + "_images"
            output_path = Path(output_file)
            new_images_dir = str(output_path.parent / f"{output_path.stem}_images")
        
        new_images_path = Path(new_images_dir)
        os.makedirs(new_images_path, exist_ok=True)
        
        # 如果有子集，创建子集目录
        if has_subsets:
            for subset in valid_subsets:
                subset_img_dir = new_images_path / subset
                os.makedirs(subset_img_dir, exist_ok=True)
        
        print(f"将创建新图像目录: {new_images_dir}")
    
    # 处理图像和标注
    image_id = 0
    annotation_id = 0
    
    # 用于跟踪每个子集的图像计数
    subset_counters = {subset: 0 for subset in valid_subsets}
    if not has_subsets:
        subset_counters["dataset"] = 0
    
    # 用于保存文件名映射
    filename_mapping = {}
    
    # 遍历每个子集（或单一数据集）
    for subset, img_dir in image_dirs.items():
        print(f"处理子集: {subset}")
        
        # 获取该子集/目录下的所有图像文件
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        
        label_dir = label_dirs.get(subset)
        
        for i, img_file in enumerate(tqdm(sorted(image_files), desc=f"处理{subset}图像")):
            # 获取图像信息
            img = Image.open(img_file)
            width, height = img.size
            
            # 原始文件名
            orig_file_name = img_file.name
            
            # 处理新图像文件（如果需要）
            if create_new_images:
                # 构建新文件名 (001.jpg, 002.jpg, ... 或 train_001.jpg, ...)
                if has_subsets:
                    new_filename = f"{subset}_{subset_counters[subset]:06d}{img_file.suffix}"
                    subset_counters[subset] += 1
                    new_img_path = Path(new_images_dir) / subset / new_filename
                else:
                    new_filename = f"{subset_counters['dataset']:06d}{img_file.suffix}"
                    subset_counters['dataset'] += 1
                    new_img_path = Path(new_images_dir) / new_filename
                
                # 复制图像到新目录
                shutil.copy2(img_file, new_img_path)
                
                # 在COCO文件中使用新文件名
                file_name = new_filename
                
                # 保存映射关系
                filename_mapping[str(img_file)] = str(new_img_path)
            else:
                # 使用原始文件名
                file_name = orig_file_name
            
            # 创建图像信息
            image_info = create_image_info(image_id, file_name, (width, height))
            coco_output["images"].append(image_info)
            
            # 找到对应的标签文件
            if label_dir:
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f.readlines():
                            if line.strip():
                                parts = line.strip().split()
                                class_id = int(parts[0])
                                # YOLO格式：class_id center_x center_y width height
                                # 所有值是归一化的[0-1]
                                center_x = float(parts[1]) * width
                                center_y = float(parts[2]) * height
                                box_width = float(parts[3]) * width
                                box_height = float(parts[4]) * height
                                
                                # 转换为COCO格式的边界框 [x, y, width, height]
                                x = center_x - (box_width / 2)
                                y = center_y - (box_height / 2)
                                
                                annotation = create_annotation(
                                    annotation_id, 
                                    image_id, 
                                    class_id, 
                                    [x, y, box_width, box_height]
                                )
                                coco_output["annotations"].append(annotation)
                                annotation_id += 1
            
            image_id += 1
    
    # 保存COCO格式JSON文件
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # 如果创建了新图像，保存文件名映射
    if create_new_images:
        mapping_file = Path(new_images_dir) / "filename_mapping.txt"
        with open(mapping_file, 'w') as f:
            for old_name, new_name in filename_mapping.items():
                f.write(f"{old_name} -> {new_name}\n")
        print(f"文件名映射已保存至: {mapping_file}")
    
    print(f"转换完成！COCO格式数据已保存至: {output_file}")
    print(f"共处理 {image_id} 张图像和 {annotation_id} 个标注")
    print(f"类别信息: {classes}")
    
    if create_new_images:
        print(f"新图像已保存至: {new_images_dir}")
    
    return output_file, new_images_dir if create_new_images else None


if __name__ == "__main__":
    # 直接在这里设置参数，无需命令行输入
    # 修改以下参数即可使用
    yolo_dir = "datasets/merged_dataset_0425_4"  # YOLO数据集目录
    output_file = "merged_dataset_0425_4_coco.json"  # 输出的COCO格式JSON文件路径
    class_file = None  # 类别文件路径（None表示自动从data.yaml读取）
    
    # 新增参数：是否创建新图像文件夹并重命名图像
    create_new_images = True  # 设置为True则创建新图像文件夹并处理重名
    new_images_dir = "datasets/coco_merged_dataset_0425_4"  # 新图像文件夹路径，可以设为None使用默认值
    
    # 可选：如果要指定类别文件，取消下面这行的注释并设置正确的路径
    # class_file = "datasets/merged_dataset_0425_4/classes.txt"
    
    convert_yolo_to_coco(yolo_dir, output_file, class_file, create_new_images, new_images_dir)