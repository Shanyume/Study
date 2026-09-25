# -*- coding: utf-8 -*-

from __future__ import print_function
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from data import load_data
import random


#本文件把训练好的CNN当作"特征提取器"：用CNN的全连接层输出作为特征，
#再交给传统分类器(SVM、随机森林)训练，与CNN自带的softmax分类头对比效果。
#注意：分类头是端到端用交叉熵训练的，特征与任务是联合优化的；
#而CNN特征+SVM的方式特征提取与分类器训练是解耦的，两者精度各有千秋。

#用SVM做分类：C是正则化参数（越大对误分类惩罚越重），kernel='rbf'径向基核可处理非线性可分
def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    #在测试集上预测，accuracy=预测正确的样本数/测试集大小
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

#用随机森林做分类：n_estimators=400棵决策树，criterion='gini'用基尼指数选分裂特征
def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

if __name__ == "__main__":
    #load data
    data, label = load_data()
    #shuffle the data
    #shuffle the data，打乱数据保证训练/测试划分随机
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    
    #前30000个样本作训练集，后12000个作测试集
    (traindata,testdata) = (data[0:30000],data[30000:])
    (trainlabel,testlabel) = (label[0:30000],label[30000:])
    #加载cnn.py训练保存的模型model.pkl
    #use origin_model to predict testdata
    origin_model = cPickle.load(open("model.pkl","rb"))
    #print(origin_model.layers)
    #原始模型（CNN+softmax分类头）在测试集上的精度
    pred_testlabel = origin_model.predict_classes(testdata,batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(" Origin_model Accuracy:",accuracy)
    #define theano funtion to get output of FC layer
    #用theano定义前向传播函数：输入是第0层的x，输出是layers[9]的输出（全连接隐藏层/Flatten，
    #具体是Flatten的256维输出还是Dense(128)的128维输出，取决于Keras版本对层的计数方式）
    get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[9].output,allow_input_downcast=False)
    #对所有42000张图提取该层特征，得到(42000, 128或256)的特征矩阵
    feature = get_feature(data)
    #train svm using FC-layer feature
    #MinMaxScaler把每个特征维缩放到[0,1]，缓解SVM对特征尺度的敏感性
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    #用CNN特征+SVM训练，并与上面原始模型、(rf函数可再跑)随机森林对比精度
    svc(feature[0:30000],label[0:30000],feature[30000:],label[30000:])
