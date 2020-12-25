# 数据准备，将所有图片放置于'./dataset/images'，所有xml标签放置于'./dataset/ori_labels'，（新建文件夹./dataset/labels）
# 运行该脚本，最终图片分类txt保存于./dataset/images'，转换为txt的标签保存于'./dataset/labels'
# classes为类别名称，依数据集更改，？？可尝试中文
# 可改变autosplit函数中weights改变训练集、验证集、测试集图片占总图片比例

import random
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
from os import getcwd

# sets = ['train', 'val', 'test']
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# abs_path = os.getcwd()

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes


def convert(size, box):   # 转换包围盒格式
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):   # 转换标签image_id为txt格式
    in_file = open('./dataset/ori_labels/%s.xml' % (image_id))
    out_file = open('./dataset/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 自动切分数据集为训练集、验证集、测试集，标签不需要划分，只需转换，见utils/datasets.py中img2label_paths(img_paths):  # 由图片路径寻找标签路径
def autosplit(path='./dataset/images', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')

    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    # Arguments
        path:       Path to images directory
        weights:    Train, val, test weights (list)
    """
    path = Path(path)  # images dir
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
    txt = ['train.txt', 'val.txt', 'test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
    for i, img in tqdm(zip(indices, files), total=n):
        if img.suffix[1:] in img_formats:   # 如果后缀属于img_formats，去除.
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file


autosplit()  # 自动分割训练集、验证集、测试集


path = Path('./dataset/ori_labels')
files = list(path.rglob('*.*'))  # 遍历路径
for item in files:
    convert_annotation(item.stem)  # 遍历将所有标签由xml转换为txt
    # item为目录 .name取文件名   .stem取去后缀文件名   .suffix取后缀
    # .endswith('.xml') 或.endswith('xml') 判断后缀是否为.xml


