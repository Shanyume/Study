# -*- coding: utf-8 -*-
"""
LibSVM/LibLinear 数据文件生成
================================

把手写数字文本图像转为 libsvm 格式的数据文件。
"""
#coding:utf-8
"""
讲文件下的所有图像写进csv文件，一行代表一张图
"""
import os
from PIL import Image
import numpy as np
import csv

#获取图像标签，以jaffe数据库为例。
# 文件名形如 <编号>_<表情>_1.tif，表情缩写取第 2 段前两个字符
def getlabel(img_name):
	face_expression = img_name.split('.')[1]   # # 去掉扩展名 .tif，得到 "编号_表情_N"
	face_expression = face_expression[0:2]     # # 取编号前两位作为类别标识（Jaffe 编号 1-7 对应 7 类表情）
	table={'HA':1,'AN':2,'SU':3,'FE':4,'DI':5,'SA':6,'NE':7}
	# 表情缩写 -> 1..7 的映射（Happiness / Anger / Surprise / Fear / Disgust / Sadness / Neutral）
	return table.get(face_expression)


f = csv.writer(open("trainlbp.csv","wb"))  # # 输出 CSV 文件：一行一张图（标签 + 灰度值）


direction = "./jaffe"  # # 图像所在目录
img_list = os.listdir(direction)  # # 列出目录下所有图像文件
for imgname in img_list:
	img = Image.open(direction+imgname)  # # 打开图像
	width,height = img.size
	data = np.empty((width*height+1))  # # 一行数据：1 个标签 + width*height 个像素

	data[0] = getlabel(imgname)  # # 第 0 列：类别标签
	img_mat = np.array(img,dtype="float")
	img_mat = img_mat.flatten()  # # 二维灰度矩阵展平为一维
	data [1:] = img_mat  # # 其余列：像素值
	#write into file
	f.writerow(data)  # # 写入一行








