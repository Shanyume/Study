# -*- coding: utf-8 -*-
"""
File:cnn.py
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py

2016.06.06更新：
这份代码是keras开发初期写的，当时keras还没有现在这么流行，文档也还没那么丰富，所以我当时写了一些简单的教程。
现在keras的API也发生了一些的变化，建议及推荐直接上keras.io看更加详细的教程。

"""
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_data
import random,cPickle
from keras.callbacks import EarlyStopping
import numpy as np

#固定随机种子，保证权重随机初始化、数据打乱等随机过程可复现
np.random.seed(1024)  # for reproducibility


#加载数据
data, label = load_data()


#label为0~9共10个类别，keras要求形式为binary class matrices,转化一下，直接调用keras提供的这个函数
#即one-hot编码：第c个类别的标签变成[0,...,1,0,...]这样的10维向量
nb_class = 10
label = np_utils.to_categorical(label, nb_class)


def create_model():
	#Sequential表示各层按add顺序串联的"堆叠式"网络
	model = Sequential()
	#第一个卷积层：4个5x5卷积核，输入是1通道28x28的灰度图，border_mode='valid'表示不补边
	#卷积后输出4张(24,24)的特征图
	model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=(1,28,28))) 
	#relu激活：max(0, x)，比tanh梯度更不容易消失，训练更快
	model.add(Activation('relu'))

	#第二个卷积层：8个3x3卷积核，输入是上一层的4张特征图，输出8张(22,22)的特征图
	model.add(Convolution2D(8,3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	#2x2最大池化：特征图尺寸减半(22,22)->(11,11)，减小参数量并提供一定平移不变性
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#第三个卷积层：16个3x3卷积核，输入8张(11,11)特征图，输出16张(9,9)的特征图
	model.add(Convolution2D(16,3, 3, border_mode='valid')) 
	model.add(Activation('relu'))
	#2x2最大池化：(9,9)->(4,4)，忽略边界
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#Flatten把16张(4,4)特征图拉平成一维向量，长度16*4*4=256，作为全连接层的输入
	model.add(Flatten())
	#全连接层（隐藏层）：128个神经元，权重用正态分布随机初始化
	model.add(Dense(128, init='normal'))
	model.add(Activation('relu'))

	#输出层：10个神经元对应10个类别，softmax输出各类别概率，argmax即预测类别
	model.add(Dense(nb_class, init='normal'))
	model.add(Activation('softmax'))
	return model


#############
#开始训练模型
##############
model = create_model()
#SGD：lr学习速率，decay权重衰减(L2正则)，momentum动量，nesterov=True使用Nesterov动量（先向前看一步再更新）
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#分类交叉熵损失：对每个样本真实类别的预测概率取-log求均值
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#打乱样本顺序，避免相邻batch来自同一类别导致梯度偏差
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
#前30000个样本作训练集，后12000个作验证集，用于训练时监控过拟合
(X_train,X_val) = (data[0:30000],data[30000:])
(Y_train,Y_val) = (label[0:30000],label[30000:])

#使用early stopping返回最佳epoch对应的model
#patience=1：验证集loss连续1个epoch不再下降就提前停止训练，防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, batch_size=100,validation_data=(X_val, Y_val),nb_epoch=5,callbacks=[early_stopping])
#把训练好的模型序列化保存到磁盘，供get_feature_map.py和cnn-svm.py加载复用
cPickle.dump(model,open("./model.pkl","wb"))
