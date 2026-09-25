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
Source:https://github.com/wepe

"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，图像大小28*28
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    #data形状(batch,通道,高,宽)：42000张28x28灰度图，第0维对应batch维度
    data = np.empty((42000,1,28,28),dtype="float32")
    #label存放每张图的文件名（去掉扩展名后的数字），即该图片对应的类别0~9
    label = np.empty((42000,),dtype="uint8")

    imgs = os.listdir("./mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("./mnist/"+imgs[i])
        #PIL读入的图像转为float32数组，取值范围0~255
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        #文件名本身带有类别编号（wepe教程的mnist目录按类别命名），取扩展名之前的数字作为标签
        label[i] = int(imgs[i].split('.')[0])
        #注意：归一化和零均值化写在循环内（作者笔误），实际效果与在循环外做一次相同
        data /= np.max(data)
        data -= np.mean(data)
    return data,label






