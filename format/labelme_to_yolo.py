# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:42:11 2022
@author: https://blog.csdn.net/suiyingy?type=blog
"""
import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
from glob import glob
 
cls2id = {'cestbon': 0, "cola": 1, "cup": 2, "ball" : 3, "basket" : 4}
 
#支持中文路径
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),flags=cv2.IMREAD_COLOR)
    return cv_img
 
def labelme2yolo_single(label_file):
    anno = json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    w0, h0 = anno['imageWidth'], anno['imageHeight']
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x = (x1 + x2) / 2 / w0
        y = (y1 + y2) / 2 / h0
        w = abs(x2 - x1) / w0
        h = abs(y2 - y1) / h0
        cid = cls2id[s['label']]
        labels.append([cid, x, y, w, h])
    return np.array(labels, dtype=np.float32)

def labelme2yolo(labelme_label_dir, save_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    yolo_label_dir = save_dir + 'labels/'
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    json_files = glob(labelme_label_dir + '*.json')
    for ijf, jf in enumerate(json_files):
        print(ijf + 1, '/', len(json_files), jf)
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels = labelme2yolo_single(jf)
        if len(labels) > 0:
            np.savetxt(yolo_label_dir + filename + '.txt', labels, fmt='%d %.6f %.6f %.6f %.6f')
    print('Completed!')
    
if __name__ == '__main__':
    root_dir = "./my_origindata/labelme_labels"
    save_dir = "./my_origindata" # 不用加 labels
    labelme2yolo(root_dir, save_dir)