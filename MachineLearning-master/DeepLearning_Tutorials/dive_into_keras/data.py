# -*- coding: utf-8 -*-
"""
Keras 数据加载
================

加载 MNIST 图像数据并归一化，供 Keras 模型使用。
旧版教程存档。
"""
#coding:utf-8
"""
Author:wepon
Code:https://github.com/wepe

File: data.py

download data here: http://pan.baidu.com/s/1qCdS6

"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，如果是将彩色图作为输入,则将1替换为3,图像大小28*28
def load_data():
	#data形状(batch,通道,高,宽)：42000张28x28灰度图，第0维对应batch维度
	data = np.empty((42000,1,28,28),dtype="float32")
	#label存放每张图对应的类别（0~9），取自文件名去掉扩展名后的数字
	label = np.empty((42000,),dtype="uint8")
	imgs = os.listdir("./mnist")
	num = len(imgs)
	for i in range(num):
		img = Image.open("./mnist/"+imgs[i])
		#PIL读入的图像转为float32数组，像素取值范围0~255
		arr = np.asarray(img,dtype="float32")
		data[i,:,:,:] = arr
		label[i] = int(imgs[i].split('.')[0])
	#归一化和零均值化
	#归一化：除以最大值，把所有像素缩放到0~1之间，加快收敛
	#零均值化：再减去数据集均值，使输入均值为0，进一步稳定训练
	data /= np.max(data)
	data -= np.mean(data)
	return data,label







