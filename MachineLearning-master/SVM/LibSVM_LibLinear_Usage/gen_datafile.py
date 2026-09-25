# -*- coding: utf-8 -*-
"""
LibSVM/LibLinear 数据文件生成
================================

把手写数字文本图像（或 Jaffe 人脸表情图像）转为 CSV 格式的数据文件，
每行代表一张图像：第一列为类别标签，其余列为展平后的像素值。

处理流程：
    1. 遍历指定目录下的所有图像文件
    2. 从文件名提取类别标签（以 Jaffe 数据库为例）
    3. 打开图像，转为灰度矩阵并展平为一维向量
    4. 将 [标签, 像素值1, 像素值2, ...] 写入 CSV 文件

Jaffe 数据库说明：
    - 包含 7 种面部表情（HA/AN/SU/FE/DI/SA/NE）
    - 文件名格式：<编号>_<表情缩写>_1.tif
    - 表情缩写映射：HA=1, AN=2, SU=3, FE=4, DI=5, SA=6, NE=7

输出格式：
    - trainlbp.csv：每行一张图，第一列为标签，其余列为像素值
    - 可直接用于 LibSVM/LibLinear 训练（需进一步转为 libsvm 格式）

依赖：PIL、numpy、csv
"""
#coding:utf-8
"""
将目录下的所有图像写进 CSV 文件，一行代表一张图
"""
import os
from PIL import Image
import numpy as np
import csv

def getlabel(img_name):
    """
    从文件名提取图像类别标签（以 Jaffe 数据库为例）

    文件名格式：<编号>_<表情缩写>_1.tif
    例如：KA.AN1.39.tif → 表情缩写 AN → 标签 2

    :param img_name: 图像文件名
    :return: 整数标签（1~7），对应 7 种表情
    """
    # 去掉扩展名 .tif，得到 "编号_表情_N"
    face_expression = img_name.split('.')[1]
    # 取编号前两位作为类别标识（Jaffe 编号 1-7 对应 7 类表情）
    face_expression = face_expression[0:2]
    # 表情缩写 -> 1..7 的映射
    # HA=Happiness, AN=Anger, SU=Surprise, FE=Fear, DI=Disgust, SA=Sadness, NE=Neutral
    table={'HA':1,'AN':2,'SU':3,'FE':4,'DI':5,'SA':6,'NE':7}
    return table.get(face_expression)


# 输出 CSV 文件：一行一张图（标签 + 灰度值）
f = csv.writer(open("trainlbp.csv","wb"))

# 图像所在目录
direction = "./jaffe"
# 列出目录下所有图像文件
img_list = os.listdir(direction)
for imgname in img_list:
    # 打开图像文件
    img = Image.open(direction+imgname)
    width,height = img.size
    # 一行数据：1 个标签 + width*height 个像素值
    data = np.empty((width*height+1))

    # 第 0 列：类别标签（从文件名提取）
    data[0] = getlabel(imgname)
    # 将图像转为浮点灰度矩阵
    img_mat = np.array(img,dtype="float")
    # 二维灰度矩阵展平为一维向量
    img_mat = img_mat.flatten()
    # 其余列：像素值（按行优先顺序展平）
    data [1:] = img_mat
    # 写入 CSV 文件的一行
    f.writerow(data)
