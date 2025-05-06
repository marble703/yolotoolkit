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


def convert_yolo_to_coco(yolo_dir, output_root, class_file=None):
    """
    将YOLO格式数据集转换为COCO格式，输出目录结构：
    output_root/
      images/{subset}/
      annotations/instances_{subset}.json
    """
    yolo_path = Path(yolo_dir)
    # 检查子集: train, val, test
    subsets = [d for d in ("train","val","test") if (yolo_path/d).exists()]
    if not subsets:
        subsets = ["dataset"]
    # 准备输出目录（标准 COCO 结构）
    root = Path(output_root)
    annotations_root = root/"annotations"
    # 创建 train2017, val2017, test2017 目录
    for s in subsets:
        (root/f"{s}2017").mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)
    
    # 读取类别信息（保持原有方式）
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
    
    # 设置类别字典
    categories = []
    for i, cls in enumerate(classes):
        categories.append({"id":i,"name":cls,"supercategory":"none"})
    
    # 对每个子集分别生成 JSON
    for subset in subsets:
        coco = {"info":{},"licenses":[{"id":1,"name":"Unknown","url":""}],"categories":categories,"images":[],"annotations":[]}
        image_id=0; ann_id=0
        # 定位子集中的 images 和 labels 目录
        subset_dir = yolo_path / subset
        if (subset_dir / "images").exists():
            img_dir = subset_dir / "images"
        elif (yolo_path / "images").exists():
            img_dir = yolo_path / "images"
        else:
            img_dir = subset_dir
        if (subset_dir / "labels").exists():
            label_dir = subset_dir / "labels"
        elif (yolo_path / "labels").exists():
            label_dir = yolo_path / "labels"
        else:
            label_dir = subset_dir
        # 收集支持的图像文件（递归查找所有子目录）
        img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            img_files.extend(sorted(img_dir.rglob(ext)))
        print(f"Processing subset '{subset}', using image directory: {img_dir}, found {len(img_files)} images")
        for img_file in img_files:
            # 图片复制
            dst = root/f"{subset}2017"/img_file.name
            try:
                shutil.copy2(img_file, dst)
                print(f"Copy {img_file} -> {dst}")
            except Exception as e:
                print(f"Failed to copy {img_file}: {e}")
            width,height=Image.open(img_file).size
            coco["images"].append(create_image_info(image_id,img_file.name,(width,height)))
            # 标注读取
            lf = label_dir/(img_file.stem+".txt")
            if lf.exists():
                for line in open(lf):
                    parts=line.split(); cid=int(parts[0]); cx=float(parts[1])*width; cy=float(parts[2])*height; w=float(parts[3])*width; h=float(parts[4])*height; x=cx-w/2; y=cy-h/2
                    coco["annotations"].append(create_annotation(ann_id,image_id,cid,[x,y,w,h]))
                    ann_id+=1
            image_id+=1
        # 保存子集 JSON
        out_file = annotations_root/f"instances_{subset}2017.json"
        with open(out_file,'w') as f:
            json.dump(coco,f,indent=2)
    return output_root


if __name__ == "__main__":
    # 直接在这里设置参数，无需命令行输入
    yolo_dir = "datasets/merged_dataset_0505_plus"  # YOLO数据集目录
    output_root = "datasets/coco_merged_dataset_0505_plus"  # 输出的COCO格式JSON文件路径
    # 使用默认类别读取方式
    convert_yolo_to_coco(yolo_dir, output_root)