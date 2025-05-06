# -*- coding: utf-8 -*-
"""
Created on 2023-04-01 9:08

@author: Fan yi ming

Func: 对于目标检测的数据增强[YOLO]（特点是数据增强后标签也要更改）
review：常用的数据增强方式；
        1.翻转：左右和上下翻转，随机翻转
        2.随机裁剪，图像缩放
        3.改变色调
        4.添加噪声

注意： boxes的标签和坐标一个是int，一个是float，存放的时候要注意处理方式。

参考：https://github.com/REN-HT/Data-Augmentation/blob/main/data_augmentation.py
"""
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFile
from PIL import ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil
from tqdm import tqdm
import concurrent.futures

random.seed(0)


class DataAugmentationOnDetection:
    def __init__(self):
        super(DataAugmentationOnDetection, self).__init__()

    # 以下的几个参数类型中，image的类型全部如下类型
    # 参数类型： image：Image.open(path)
    def resize_keep_ratio(self, image, boxes, target_size):
        """
            参数类型： image：Image.open(path)， boxes:Tensor， target_size:int
            功能：将图像缩放到size尺寸，调整相应的boxes,同时保持长宽比（最长的边是target size
        """
        old_size = image.size[0:2]  # 原始图像大小
        # 取最小的缩放比例
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
        new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
        # boxes 不用变化，因为是等比例变化
        return image.resize(new_size, Image.BILINEAR), boxes

    def resizeDown_keep_ratio(self, image, boxes, target_size):
        """ 与上面的函数功能类似，但它只降低图片的尺寸，不会扩大图片尺寸"""
        old_size = image.size[0:2]  # 原始图像大小
        # 取最小的缩放比例
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
        ratio = min(ratio, 1)
        new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小

        # boxes 不用变化，因为是等比例变化
        return image.resize(new_size, Image.BILINEAR), boxes

    def resize(self, img, boxes, size):
        # ---------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
        # ---------------------------------------------------------
        return img.resize((size, size), Image.BILINEAR), boxes

    def random_flip_horizon(self, img, boxes, h_rate=1):
        # -------------------------------------
        # 固定水平翻转
        # -------------------------------------
        transform = transforms.RandomHorizontalFlip(p=1)
        img = transform(img)
        if len(boxes) > 0:
            x = 1 - boxes[:, 1]
            boxes[:, 1] = x
        return img, boxes

    def random_flip_vertical(self, img, boxes, v_rate=1):
        # 固定垂直翻转
        transform = transforms.RandomVerticalFlip(p=1)
        img = transform(img)
        if len(boxes) > 0:
            y = 1 - boxes[:, 2]
            boxes[:, 2] = y
        return img, boxes

    def center_crop(self, img, boxes, target_size=None):
        # -------------------------------------
        # 中心裁剪 ，裁剪成 (size, size) 的正方形, 仅限图形，w,h
        # 这里用比例是很难算的，转成x1,y1, x2, y2格式来计算
        # -------------------------------------
        w, h = img.size
        size = min(w, h)
        if len(boxes) > 0:
            # 转换到xyxy格式
            label = boxes[:, 0].reshape([-1, 1])
            x_, y_, w_, h_ = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            x1 = (w * x_ - 0.5 * w * w_).reshape([-1, 1])
            y1 = (h * y_ - 0.5 * h * h_).reshape([-1, 1])
            x2 = (w * x_ + 0.5 * w * w_).reshape([-1, 1])
            y2 = (h * y_ + 0.5 * h * h_).reshape([-1, 1])
            boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)
            # 边框转换
            if w > h:
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - (w - h) / 2
            else:
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - (h - w) / 2
            in_boundary = [i for i in range(boxes_xyxy.shape[0])]
            for i in range(boxes_xyxy.shape[0]):
                # 判断x是否超出界限
                if (boxes_xyxy[i, 0] < 0 and boxes_xyxy[i, 2] < 0) or (boxes_xyxy[i, 0] > size and boxes_xyxy[i, 2] > size):
                    in_boundary.remove(i)
                # 判断y是否超出界限
                elif (boxes_xyxy[i, 1] < 0 and boxes_xyxy[i, 3] < 0) or (boxes_xyxy[i, 1] > size and boxes_xyxy[i, 3] > size):
                    in_boundary.append(i)
            boxes_xyxy = boxes_xyxy[in_boundary]
            boxes = boxes_xyxy.clamp(min=0, max=size).reshape([-1, 4])  # 压缩到固定范围
            label = label[in_boundary]
            # 转换到YOLO格式
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            xc = ((x1 + x2) / (2 * size)).reshape([-1, 1])
            yc = ((y1 + y2) / (2 * size)).reshape([-1, 1])
            wc = ((x2 - x1) / size).reshape([-1, 1])
            hc = ((y2 - y1) / size).reshape([-1, 1])
            boxes = torch.cat([xc, yc, wc, hc], dim=1)
        # 图像转换
        transform = transforms.CenterCrop(size)
        img = transform(img)
        if target_size:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        if len(boxes) > 0:
            return img, torch.cat([label.reshape([-1, 1]), boxes], dim=1)
        else:
            return img, boxes

    def gaussian_blur(self, img, boxes, radius=2):
        """
        对图像应用高斯模糊
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            radius: 模糊半径
        """
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
        return blurred_img, boxes

    # ------------------------------------------------------
    # 以下img皆为Tensor类型
    # ------------------------------------------------------

    def random_bright(self, img, u=120, p=1):
        # -------------------------------------
        # 随机亮度变换
        # -------------------------------------
        if np.random.random() < p:
            alpha=np.random.uniform(-u, u)/255
            img += alpha
            img=img.clamp(min=0.0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5, p=1):
        # -------------------------------------
        # 随机增强对比度
        # -------------------------------------
        if np.random.random() < p:
            alpha=np.random.uniform(lower, upper)
            img*=alpha
            img=img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img,lower=0.5, upper=1.5, p=1):
        # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
        if np.random.random() < p:
            alpha=np.random.uniform(lower, upper)
            img[1]=img[1]*alpha
            img[1]=img[1].clamp(min=0,max=1.0)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        noise=torch.normal(mean,std,img.shape)
        img+=noise
        img=img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()/5 + 0.7
        img[noise[:,:,:]>alpha]=1.0
        return img

    def add_pepper_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()/5 + 0.7
        img[noise[:, :, :]>alpha]=0
        return img

    def stretch_image(self, img, boxes, x_rate=1.0, y_rate=1.0):
        """
        在 x、y 方向上拉伸或压缩图像
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            x_rate: x方向的缩放比例
            y_rate: y方向的缩放比例
        """
        w, h = img.size
        new_w = int(w * x_rate)
        new_h = int(h * y_rate)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        if len(boxes) > 0:
            # 调整框的坐标
            boxes = boxes.clone()
            # YOLO格式的坐标是相对坐标，在非等比例缩放时需要调整中心点位置和宽高
            # x center
            boxes[:, 1] = boxes[:, 1] * w / new_w
            # y center
            boxes[:, 2] = boxes[:, 2] * h / new_h
            # width
            boxes[:, 3] = boxes[:, 3] * w / new_w
            # height
            boxes[:, 4] = boxes[:, 4] * h / new_h
            
        return img, boxes

    def rotate_image_and_boxes(self, img, boxes, angle):
        """
        旋转图像和边界框
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            angle: 旋转角度，顺时针为正
        """
        # 旋转图像
        rotated_img = img.rotate(-angle, expand=True, resample=Image.BILINEAR)
        
        if len(boxes) == 0:
            return rotated_img, boxes
            
        # 获取原始图像和旋转后图像的尺寸
        w, h = img.size
        new_w, new_h = rotated_img.size
        
        # 将YOLO格式的框转换为x1,y1,x2,y2格式
        boxes_copy = boxes.clone()
        label = boxes_copy[:, 0].reshape([-1, 1])
        x_center, y_center = boxes_copy[:, 1], boxes_copy[:, 2]
        width, height = boxes_copy[:, 3], boxes_copy[:, 4]
        
        # 计算边界框的四个角点坐标（相对坐标）
        x1 = (x_center - width/2)
        y1 = (y_center - height/2)
        x2 = (x_center + width/2)
        y2 = (y_center + height/2)
        
        # 将相对坐标转换为绝对坐标
        x1 = x1 * w
        y1 = y1 * h
        x2 = x2 * w
        y2 = y2 * h
        
        # 计算旋转中心（原图中心）
        cx, cy = w/2, h/2
        
        # 旋转四个角点
        corners = []
        for i in range(len(boxes)):
            # 获取该框的四个角点
            box_corners = [
                [x1[i], y1[i]],  # 左上
                [x2[i], y1[i]],  # 右上
                [x2[i], y2[i]],  # 右下
                [x1[i], y2[i]]   # 左下
            ]
            
            # 旋转每个角点
            rotated_corners = []
            for x, y in box_corners:
                # 移到原点
                x -= cx
                y -= cy
                
                # 旋转
                angle_rad = np.radians(angle)
                x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # 移回并调整到新的图像尺寸
                x_rot += new_w/2
                y_rot += new_h/2
                
                rotated_corners.append([x_rot, y_rot])
            
            corners.append(rotated_corners)
        
        # 计算旋转后的边界框
        rotated_boxes = []
        for i, box_corners in enumerate(corners):
            # 找出最小的外接矩形
            min_x = min(corner[0] for corner in box_corners)
            min_y = min(corner[1] for corner in box_corners)
            max_x = max(corner[0] for corner in box_corners)
            max_y = max(corner[1] for corner in box_corners)
            
            # 确保边界框在图像范围内
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(new_w, max_x)
            max_y = min(new_h, max_y)
            
            # 如果旋转后的框太小，则跳过
            if max_x - min_x < 1 or max_y - min_y < 1:
                continue
                
            # 转换回YOLO格式（相对坐标）
            x_center = (min_x + max_x) / 2 / new_w
            y_center = (min_y + max_y) / 2 / new_h
            width = (max_x - min_x) / new_w
            height = (max_y - min_y) / new_h
            
            rotated_boxes.append([label[i].item(), x_center, y_center, width, height])
        
        if not rotated_boxes:
            return rotated_img, torch.tensor([])
            
        return rotated_img, torch.tensor(rotated_boxes)

    def random_occlusion(self, img, boxes, num_rectangles=3, size_range=(0.1, 0.3), color='random'):
        """
        使用纯白色或黑色随机遮挡图像的一部分
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            num_rectangles: 遮挡矩形的数量
            size_range: 遮挡矩形大小比例的范围，相对于图像尺寸
            color: 遮挡颜色，'white'、'black'或'random'
        """
        img_copy = img.copy()
        w, h = img_copy.size
        draw = ImageDraw.Draw(img_copy)
        
        for _ in range(num_rectangles):
            # 随机确定矩形的尺寸
            rect_w = int(w * random.uniform(size_range[0], size_range[1]))
            rect_h = int(h * random.uniform(size_range[0], size_range[1]))
            
            # 随机确定矩形的位置
            left = random.randint(0, w - rect_w)
            top = random.randint(0, h - rect_h)
            right = left + rect_w
            bottom = top + rect_h
            
            # 随机确定颜色
            if color == 'random':
                fill_color = random.choice([(255, 255, 255), (0, 0, 0)])  # 白色或黑色
            elif color == 'white':
                fill_color = (255, 255, 255)  # 白色
            else:  # 黑色
                fill_color = (0, 0, 0)  # 黑色
                
            # 绘制矩形
            draw.rectangle([left, top, right, bottom], fill=fill_color)
            
        # 边界框不需要修改，因为目标的位置没有变化，只是被部分遮挡
        return img_copy, boxes

    def shift_box_content(self, img, boxes, direction='right', shift_ratio=0.2):
        """
        将标注框内的图像内容向指定方向移动一小段距离
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            direction: 移动方向，'left', 'right', 'up', 'down'
            shift_ratio: 移动距离占框尺寸的比例
        """
        if len(boxes) == 0:
            return img, boxes
            
        img_copy = img.copy()
        w, h = img_copy.size
        img_array = np.array(img_copy)
        
        # 处理每个边界框
        for i in range(len(boxes)):
            # 提取框信息（YOLO格式：class, x_center, y_center, width, height）
            label, x_center, y_center, width, height = boxes[i]
            
            # 计算框的绝对坐标
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 提取框内图像区域
            box_region = img_array[y1:y2, x1:x2].copy()
            
            # 计算移动距离
            if direction in ['left', 'right']:
                shift_distance = int(width * w * shift_ratio)
            else:  # 'up' 或 'down'
                shift_distance = int(height * h * shift_ratio)
                
            # 根据方向移动图像内容
            if direction == 'right':
                # 右移：右侧部分留空（黑色），左侧部分移入
                if (shift_distance >= box_region.shape[1]):
                    shift_distance = box_region.shape[1] - 1
                shifted_region = np.zeros_like(box_region)
                shifted_region[:, shift_distance:] = box_region[:, :-shift_distance]
            elif direction == 'left':
                # 左移：左侧部分留空（黑色），右侧部分移入
                if (shift_distance >= box_region.shape[1]):
                    shift_distance = box_region.shape[1] - 1
                shifted_region = np.zeros_like(box_region)
                shifted_region[:, :-shift_distance] = box_region[:, shift_distance:]
            elif direction == 'down':
                # 下移：底部留空（黑色），上部分移入
                if (shift_distance >= box_region.shape[0]):
                    shift_distance = box_region.shape[0] - 1
                shifted_region = np.zeros_like(box_region)
                shifted_region[shift_distance:, :] = box_region[:-shift_distance, :]
            elif direction == 'up':
                # 上移：顶部留空（黑色），下部分移入
                if (shift_distance >= box_region.shape[0]):
                    shift_distance = box_region.shape[0] - 1
                shifted_region = np.zeros_like(box_region)
                shifted_region[:-shift_distance, :] = box_region[shift_distance:, :]
            
            # 将移动后的区域放回原图
            img_array[y1:y2, x1:x2] = shifted_region
            
        # 将修改后的图像数组转换回PIL.Image
        return Image.fromarray(img_array), boxes

    def rotate_except_boxes(self, img, boxes, angle=30):
        """
        旋转整张图像，但保持标注框区域不旋转
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            angle: 旋转角度，顺时针为正
        """
        if len(boxes) == 0:
            # 如果没有边界框，直接旋转整张图像
            rotated_img = img.rotate(-angle, expand=True, resample=Image.BILINEAR)
            return rotated_img, boxes
            
        # 旋转整张图像
        rotated_img = img.rotate(-angle, expand=True, resample=Image.BILINEAR)
        
        # 获取原始图像和旋转后图像的尺寸
        w, h = img.size
        new_w, new_h = rotated_img.size
        
        # 旋转图像的中心点
        center_x, center_y = w/2, h/2
        new_center_x, new_center_y = new_w/2, new_h/2
        
        # 将旋转图像转换为numpy数组
        rotated_array = np.array(rotated_img)
        orig_array = np.array(img)
        
        # 处理每个边界框
        for i in range(len(boxes)):
            # 提取框信息（YOLO格式：class, x_center, y_center, width, height）
            label, x_center, y_center, width, height = boxes[i]
            
            # 计算框在原图中的绝对坐标
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 计算边界框的四个角点坐标
            corners = [
                [x1, y1],  # 左上
                [x2, y1],  # 右上
                [x2, y2],  # 右下
                [x1, y2]   # 左下
            ]
            
            # 旋转角点到新图像中
            rotated_corners = []
            for x, y in corners:
                # 移到原点
                x -= center_x
                y -= center_y
                
                # 旋转
                angle_rad = np.radians(angle)
                x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                # 移回并调整到新的图像尺寸
                x_rot += new_center_x
                y_rot += new_center_y
                
                rotated_corners.append([int(x_rot), int(y_rot)])
            
            # 在旋转后的图像上计算最小外接矩形
            min_x = max(0, min(corner[0] for corner in rotated_corners))
            min_y = max(0, min(corner[1] for corner in rotated_corners))
            max_x = min(new_w-1, max(corner[0] for corner in rotated_corners))
            max_y = min(new_h-1, max(corner[1] for corner in rotated_corners))
            
            # 获取旋转后矩形的宽高（确保是整数）
            rotated_width = int(max_x - min_x)
            rotated_height = int(max_y - min_y)
            
            # 如果旋转后的区域太小，则跳过
            if rotated_width < 1 or rotated_height < 1:
                continue
                
            # 提取原始边界框区域，并调整尺寸以适应旋转后的区域
            box_region = orig_array[y1:y2, x1:x2]
            if box_region.size == 0:  # 检查是否为空数组
                continue
                
            # 确保宽高至少为1
            rotated_width = max(1, rotated_width)
            rotated_height = max(1, rotated_height)
            
            try:
                box_region_resized = np.array(Image.fromarray(box_region).resize((rotated_width, rotated_height), Image.BILINEAR))
                
                # 将原始图像内容放置到旋转后的位置
                rotated_array[min_y:max_y, min_x:max_x] = box_region_resized
            except (ValueError, TypeError) as e:
                print(f"跳过不兼容的区域调整: {e}, 尺寸: ({rotated_width}, {rotated_height})")
                continue
        
        # 将修改后的图像数组转换回PIL.Image
        result_img = Image.fromarray(rotated_array)
        
        # 调整边界框坐标以适应新的图像尺寸
        adjusted_boxes = []
        for i in range(len(boxes)):
            label, x_center, y_center, width, height = boxes[i]
            
            # 计算框在原图中的绝对坐标
            x1 = (x_center - width/2) * w
            y1 = (y_center - height/2) * h
            x2 = (x_center + width/2) * w
            y2 = (y_center + height/2) * h
            
            # 旋转中心点
            angle_rad = np.radians(angle)
            x_rot = (x1 + x2) / 2 - center_x
            y_rot = (y1 + y2) / 2 - center_y
            
            new_center_x_abs = x_rot * np.cos(angle_rad) - y_rot * np.sin(angle_rad) + new_center_x
            new_center_y_abs = x_rot * np.sin(angle_rad) + y_rot * np.cos(angle_rad) + new_center_y
            
            # 转换回相对坐标
            new_x_center = new_center_x_abs / new_w
            new_y_center = new_center_y_abs / new_h
            new_width = width * w / new_w
            new_height = height * h / new_h
            
            # 确保坐标在[0,1]范围内
            new_x_center = max(0, min(1, new_x_center))
            new_y_center = max(0, min(1, new_y_center))
            new_width = min(new_width, 2 * min(new_x_center, 1 - new_x_center))
            new_height = min(new_height, 2 * min(new_y_center, 1 - new_y_center))
            
            adjusted_boxes.append([label.item(), new_x_center, new_y_center, new_width, new_height])
        
        return result_img, torch.tensor(adjusted_boxes)

    def add_horizontal_occlusion(self, img, boxes, height_ratio=0.2, color='black'):
        """
        在标注框中添加横向遮挡条
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            height_ratio: 遮挡条高度占标注框高度的比例
            color: 遮挡颜色，'white'或'black'
        """
        if len(boxes) == 0:
            return img, boxes
            
        img_copy = img.copy()
        w, h = img_copy.size
        draw = ImageDraw.Draw(img_copy)
        
        for i in range(len(boxes)):
            # 提取框信息（YOLO格式：class, x_center, y_center, width, height）
            label, x_center, y_center, width, height = boxes[i]
            
            # 计算框的绝对坐标
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 计算遮挡条的高度
            occlusion_height = int((y2 - y1) * height_ratio)
            if occlusion_height < 1:
                occlusion_height = 1
            
            # 计算遮挡条的位置（居中放置）
            occlusion_y = y1 + (y2 - y1) // 2 - occlusion_height // 2
            
            # 确定颜色
            if color == 'white':
                fill_color = (255, 255, 255)  # 白色
            else:  # 黑色
                fill_color = (0, 0, 0)  # 黑色
                
            # 绘制遮挡条
            draw.rectangle([x1, occlusion_y, x2, occlusion_y + occlusion_height], fill=fill_color)
            
        # 边界框不需要修改，因为目标的位置没有变化，只是被部分遮挡
        return img_copy, boxes
        
    def add_vertical_occlusion(self, img, boxes, width_ratio=0.2, color='black'):
        """
        在标注框中添加纵向遮挡条
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            width_ratio: 遮挡条宽度占标注框宽度的比例
            color: 遮挡颜色，'white'或'black'
        """
        if len(boxes) == 0:
            return img, boxes
            
        img_copy = img.copy()
        w, h = img.copy().size
        draw = ImageDraw.Draw(img_copy)
        
        for i in range(len(boxes)):
            # 提取框信息（YOLO格式：class, x_center, y_center, width, height）
            label, x_center, y_center, width, height = boxes[i]
            
            # 计算框的绝对坐标
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 计算遮挡条的宽度
            occlusion_width = int((x2 - x1) * width_ratio)
            if occlusion_width < 1:
                occlusion_width = 1
            
            # 计算遮挡条的位置（居中放置）
            occlusion_x = x1 + (x2 - x1) // 2 - occlusion_width // 2
            
            # 确定颜色
            if color == 'white':
                fill_color = (255, 255, 255)  # 白色
            else:  # 黑色
                fill_color = (0, 0, 0)  # 黑色
                
            # 绘制遮挡条
            draw.rectangle([occlusion_x, y1, occlusion_x + occlusion_width, y2], fill=fill_color)
            
        # 边界框不需要修改，因为目标的位置没有变化，只是被部分遮挡
        return img_copy, boxes

    def add_cross_occlusion(self, img, boxes, width_ratio=0.15, height_ratio=0.15, color='black'):
        """
        在标注框中添加十字形遮挡（横向+纵向）
        Args:
            img: PIL.Image 类型
            boxes: Tensor 类型，yolo格式的框
            width_ratio: 纵向遮挡条宽度占标注框宽度的比例
            height_ratio: 横向遮挡条高度占标注框高度的比例
            color: 遮挡颜色，'white'或'black'
        """
        # 先添加横向遮挡
        img_h_occluded, _ = self.add_horizontal_occlusion(img, boxes, height_ratio, color)
        # 再添加纵向遮挡
        return self.add_vertical_occlusion(img_h_occluded, boxes, width_ratio, color)

    def adjust_gamma(self, img, gamma=1.0):
        """
        调整图像的伽马值，实现增强曝光或降低曝光的效果
        Args:
            img: PIL.Image 类型
            gamma: 伽马值，小于1时增强曝光(图像变亮)，大于1时降低曝光(图像变暗)
        """
        # 将PIL图像转换为numpy数组进行处理
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # 应用伽马变换
        img_np = np.power(img_np, gamma)
        
        # 确保值在[0,1]范围内
        img_np = np.clip(img_np, 0, 1)
        
        # 转回uint8并重新创建PIL图像
        img_np = (img_np * 255).astype(np.uint8)
        adjusted_img = Image.fromarray(img_np)
        
        # 伽马调整只改变像素值，不会影响边界框坐标
        return adjusted_img


def plot_pics(img, boxes):
    # 显示图像和候选框，img是Image.Open()类型, boxes是Tensor类型
    plt.imshow(img)
    label_colors = [(213, 110, 89)]
    w, h = img.size
    for i in range(boxes.shape[0]):
        box = boxes[i, 1:]
        xc, yc, wc, hc = box
        x = w * xc - 0.5 * w * wc
        y = h * yc - 0.5 * h * hc
        box_w, box_h = w * wc, h * hc
        plt.gca().add_patch(plt.Rectangle(xy=(x, y), width=box_w, height=box_h,
                                          edgecolor=[c / 255 for c in label_colors[0]],
                                          fill=False, linewidth=2))
    plt.show()

def get_image_list(image_path):
    # 根据图片文件，查找所有图片并返回列表
    files_list = []
    for root, sub_dirs, files in os.walk(image_path):
        for special_file in files:
            special_file = special_file[0: len(special_file)]
            files_list.append(special_file)
    return files_list

def get_label_file(label_path, image_name):
    # 根据图片信息，查找对应的label

    fname = os.path.join(label_path, os.path.splitext(image_name)[0]+".txt")
    data2 = []
    if not os.path.exists(fname):
        return data2
    if os.path.getsize(fname) == 0:
        return data2
    else:
        with open(fname, 'r', encoding='utf-8') as infile:
            # 读取并转换标签
            for line in infile:
                data_line = line.strip("\n").split()
                data2.append([float(i) for i in data_line])
    return data2


def save_Yolo(img, boxes, save_path, prefix, image_name):
    # img: 需要时Image类型的数据， prefix 前缀
    # 将结果保存到save path指示的路径中
    if not os.path.exists(save_path) or \
            not os.path.exists(os.path.join(save_path, "images")):
        os.makedirs(os.path.join(save_path, "images"))
        os.makedirs(os.path.join(save_path, "labels"))
    try:
        img.save(os.path.join(save_path, "images", prefix + image_name))
        with open(os.path.join(save_path, "labels", prefix + os.path.splitext(image_name)[0] + ".txt"), 'w', encoding="utf-8") as f:
            if len(boxes) > 0:  # 判断是否为空
                # 写入新的label到文件中
                for data in boxes:
                    str_in = ""
                    for i, a in enumerate(data):
                        if i == 0:
                            str_in += str(int(a))
                        else:
                            str_in += " " + str(float(a))
                    f.write(str_in + '\n')
    except:
        print("ERROR: ", image_name, " is bad.")


def runAugumentation(dataset_path, save_path, max_workers=4):
    """
    对YOLO格式数据集进行数据增强，使用并发处理多张图片
    Args:
        dataset_path: YOLO数据集根目录，需包含images和labels文件夹
        save_path: 增强后的数据集保存路径
        max_workers: 最大并发进程数
    """
    image_path = os.path.join(dataset_path, "images")
    label_path = os.path.join(dataset_path, "labels")
    
    if not os.path.exists(dataset_path):
        print(f"错误：数据集路径 {dataset_path} 不存在")
        return
    if not os.path.exists(image_path):
        print(f"错误：图像文件夹 {image_path} 不存在")
        return
    if not os.path.exists(label_path):
        print(f"错误：标签文件夹 {label_path} 不存在")
        return

    # 复制 data.yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(yaml_path):
        os.makedirs(save_path, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(save_path, "data.yaml"))
        print(f"已复制 data.yaml 到输出目录")

    # 创建输出目录
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)

    image_list = get_image_list(image_path)
    total_augmentations = len(image_list)
    
    print(f"找到 {total_augmentations} 张图像，开始数据增强...")
    
    # 创建数据增强对象
    DAD = DataAugmentationOnDetection()
    
    # 保存原始图像到输出目录并加载所有图像数据
    print("正在复制原始图像到输出目录...")
    image_data = []
    for idx, image_name in enumerate(image_list, 1):
        img = Image.open(os.path.join(image_path, image_name))
        boxes = get_label_file(label_path, image_name)
        boxes = torch.tensor(boxes) if boxes else torch.tensor([])
        
        # 保存原始图像
        save_Yolo(img, boxes, save_path, "", image_name)
        
        # 只有当图像有标注框时才收集用于增强
        if len(boxes) > 0:
            image_data.append((img, boxes, image_name))
            
    print(f"共有 {len(image_data)} 张带标注框的图像需要增强处理")
    
    # 使用ProcessPoolExecutor进行并发处理
    print("开始并发数据增强处理...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务
        futures = []
        for img, boxes, image_name in image_data:
            futures.append(
                executor.submit(process_single_image, img, boxes, save_path, image_name, DAD)
            )
        
        # 显示进度条
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="数据增强进度")):
            try:
                # 获取任务结果（如果有任何异常会在这里抛出）
                future.result()
            except Exception as e:
                print(f"处理图像时发生错误: {e}")
    
    total_files = len(os.listdir(os.path.join(save_path, "images")))
    print(f"\n数据增强完成！总共生成 {total_files} 张图像。")

def process_single_image(img, boxes, save_path, image_name, DAD):
    """
    对单张图片进行数据增强处理
    Args:
        img: 原始图像 (PIL.Image)
        boxes: 边界框 (Tensor)
        save_path: 保存路径
        image_name: 图像文件名
        DAD: DataAugmentationOnDetection对象
    """
    # 只有当图像有标注框时才进行增强
    if len(boxes) == 0:
        return
        
    # 1. 水平翻转 (固定翻转)
    flipped_img, flipped_boxes = DAD.random_flip_horizon(img.copy(), boxes.clone())
    save_Yolo(flipped_img, flipped_boxes, save_path, "h_flip_", image_name)
        
    # 2. 垂直翻转 (固定翻转)
    flipped_img, flipped_boxes = DAD.random_flip_vertical(img.copy(), boxes.clone())
    save_Yolo(flipped_img, flipped_boxes, save_path, "v_flip_", image_name)
    
    # 3. 拉伸图像 (横向拉伸)
    stretched_img, stretched_boxes = DAD.stretch_image(img.copy(), boxes.clone(), x_rate=1.2, y_rate=1.0)
    save_Yolo(stretched_img, stretched_boxes, save_path, "stretch_x_", image_name)
    
    # 4. 拉伸图像 (纵向拉伸)
    stretched_img, stretched_boxes = DAD.stretch_image(img.copy(), boxes.clone(), x_rate=1.0, y_rate=1.2)
    save_Yolo(stretched_img, stretched_boxes, save_path, "stretch_y_", image_name)
    
    # 5. 亮度和对比度变化 (对tensor进行操作)
    img_tensor = transforms.ToTensor()(img)
    
    # 亮度增强
    bright_tensor = DAD.random_bright(img_tensor.clone(), u=50, p=1.0)
    bright_img = transforms.ToPILImage()(bright_tensor)
    save_Yolo(bright_img, boxes, save_path, "bright_", image_name)
    
    # 对比度增强
    contrast_tensor = DAD.random_contrast(img_tensor.clone(), lower=1.2, upper=1.5, p=1.0)
    contrast_img = transforms.ToPILImage()(contrast_tensor)
    save_Yolo(contrast_img, boxes, save_path, "contrast_", image_name)
    
    # 添加噪声
    noise_tensor = DAD.add_salt_noise(img_tensor.clone())
    noise_img = transforms.ToPILImage()(noise_tensor)
    save_Yolo(noise_img, boxes, save_path, "noise_", image_name)
    
    # 6. 高斯模糊
    blurred_img, blurred_boxes = DAD.gaussian_blur(img.copy(), boxes.clone(), radius=2)
    save_Yolo(blurred_img, blurred_boxes, save_path, "gaussian_", image_name)
    
    # 7. 随机白色遮挡
    occluded_img, occluded_boxes = DAD.random_occlusion(img.copy(), boxes.clone(), num_rectangles=2, size_range=(0.1, 0.2), color='white')
    save_Yolo(occluded_img, occluded_boxes, save_path, "occ_w_", image_name)
    
    # 8. 随机黑色遮挡
    occluded_img, occluded_boxes = DAD.random_occlusion(img.copy(), boxes.clone(), num_rectangles=2, size_range=(0.1, 0.2), color='black')
    save_Yolo(occluded_img, occluded_boxes, save_path, "occ_b_", image_name)
    
    # 9. 随机黑白混合遮挡
    occluded_img, occluded_boxes = DAD.random_occlusion(img.copy(), boxes.clone(), num_rectangles=3, size_range=(0.05, 0.15), color='random')
    save_Yolo(occluded_img, occluded_boxes, save_path, "occ_mix_", image_name)
    
    # 14. 添加标注框横向遮挡
    horiz_occluded_img, horiz_occluded_boxes = DAD.add_horizontal_occlusion(img.copy(), boxes.clone(), height_ratio=0.2, color='black')
    save_Yolo(horiz_occluded_img, horiz_occluded_boxes, save_path, "h_occ_b_", image_name)
    
    # 15. 添加标注框横向遮挡（白色）
    horiz_occluded_img, horiz_occluded_boxes = DAD.add_horizontal_occlusion(img.copy(), boxes.clone(), height_ratio=0.2, color='white')
    save_Yolo(horiz_occluded_img, horiz_occluded_boxes, save_path, "h_occ_w_", image_name)
    
    # 16. 添加标注框纵向遮挡
    vert_occluded_img, vert_occluded_boxes = DAD.add_vertical_occlusion(img.copy(), boxes.clone(), width_ratio=0.2, color='black')
    save_Yolo(vert_occluded_img, vert_occluded_boxes, save_path, "v_occ_b_", image_name)
    
    # 17. 添加标注框纵向遮挡（白色）
    vert_occluded_img, vert_occluded_boxes = DAD.add_vertical_occlusion(img.copy(), boxes.clone(), width_ratio=0.2, color='white')
    save_Yolo(vert_occluded_img, vert_occluded_boxes, save_path, "v_occ_w_", image_name)
    
    # 18. 添加标注框十字形遮挡
    cross_occluded_img, cross_occluded_boxes = DAD.add_cross_occlusion(img.copy(), boxes.clone(), width_ratio=0.15, height_ratio=0.15, color='black')
    save_Yolo(cross_occluded_img, cross_occluded_boxes, save_path, "cross_occ_b_", image_name)
    
    # 19. 添加标注框十字形遮挡（白色）
    cross_occluded_img, cross_occluded_boxes = DAD.add_cross_occlusion(img.copy(), boxes.clone(), width_ratio=0.15, height_ratio=0.15, color='white')
    save_Yolo(cross_occluded_img, cross_occluded_boxes, save_path, "cross_occ_w_", image_name)

    # 20. 增强曝光 (小伽马值，图像变亮)
    bright_gamma_img = DAD.adjust_gamma(img.copy(), gamma=0.7)
    save_Yolo(bright_gamma_img, boxes, save_path, "gamma_bright_", image_name)
    
    # 21. 降低曝光 (大伽马值，图像变暗)
    dark_gamma_img = DAD.adjust_gamma(img.copy(), gamma=1.5)
    save_Yolo(dark_gamma_img, boxes, save_path, "gamma_dark_", image_name)
    
    # 22. 更强曝光 (更小的伽马值)
    very_bright_gamma_img = DAD.adjust_gamma(img.copy(), gamma=0.5)
    save_Yolo(very_bright_gamma_img, boxes, save_path, "gamma_very_bright_", image_name)
    
    # 23. 更暗曝光 (更大的伽马值) 
    very_dark_gamma_img = DAD.adjust_gamma(img.copy(), gamma=2.0)
    save_Yolo(very_dark_gamma_img, boxes, save_path, "gamma_very_dark_", image_name)

if __name__ == '__main__':
    # 图像和标签文件夹
    dataset_path = "./my_origindata_0505"    # YOLO数据集根目录
    save_path = "./my_origindata_0505/Augumentation"    # 结果保存位置路径

    # 运行
    runAugumentation(dataset_path, save_path)