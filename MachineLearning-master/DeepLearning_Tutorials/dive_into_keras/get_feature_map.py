#coding=utf-8

"""
Author:wepon
Code:https://github.com/wepe

File: get_feature_map.py
	1.  visualize feature map of Convolution Layer, Fully Connected layer
	2.  rewrite the code so you can treat CNN as feature extractor, see file: cnn-svm.py

--
2016.06.06更新：
keras的API已经发生变化，现在可视化特征图可以直接调用接口，具体请参考：http://keras.io/visualization/


"""
from __future__ import print_function
import cPickle,theano
from data import load_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#load the saved model
#加载cnn.py训练并保存好的模型model.pkl
model = cPickle.load(open("model.pkl","rb"))

#define theano funtion to get output of  FC layer
#定义一个theano计算函数：输入是网络第0层(输入层)的x，输出是第11层(全连接隐藏层)的输出，
#即用训练好的CNN做前向传播，得到全连接层的特征向量，可看作CNN提取的特征
get_feature = theano.function([model.layers[0].input],model.layers[11].output,allow_input_downcast=False) 

#define theano funtion to get output of  first Conv layer 
#同理，输出改为第2层(第一个卷积层+relu后)的输出，即4张24x24的特征图
get_featuremap = theano.function([model.layers[0].input],model.layers[2].output,allow_input_downcast=False) 


#加载MNIST数据（只需前几张用于可视化）
data, label = load_data()

#可视化全连接层的特征：把前10张图的FC层输出拼成矩阵画出来
# visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images
feature = get_feature(data[0:10])  #visualize these images's FC-layer feature
plt.imshow(feature,cmap = cm.Greys_r)
plt.show()

#可视化第一个卷积层的特征图
#visualize feature map of Convolution Layer
#第一个卷积层有4个卷积核即4张特征图，num_fmap即要可视化几张特征图
num_fmap = 4	#number of feature map
for i in range(num_fmap):
	featuremap = get_featuremap(data[0:10])
	#featuremap[0]是第1张图的4张特征图(4,24,24)，取第i张画出来
	plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
	plt.show()
