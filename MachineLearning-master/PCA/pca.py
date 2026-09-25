# -*- coding: utf-8 -*-
"""
PCA 主成分分析
================

PCA 通过对协方差矩阵做特征分解，找到方差最大的方向作为主成分，
实现数据降维。

- zeroMean：数据中心化
- pca：按累计方差贡献率选择前 n 个主成分
- percent2n：根据贡献率计算需要保留的特征值个数

依赖：numpy
"""
import numpy as np

# 根据要求的累计方差贡献率，求需要保留的特征值个数 n
def percent2n(eigVals, percent):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmp=0
    num=0
    for i in sortArray:
        tmp+=i
        num+=1
        if tmp>=arraySum*percent:
            return num
# 数据中心化：每个特征减去均值，使均值为 0
def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal

def pca(dataMat, percent=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    
        
    eigVals,eigVects=np.linalg.eig(np.asmatrix(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    n=percent2n(eigVals,percent)                  #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)              #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]     #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]          #最大的n个特征值对应的特征向量
    lowDDataMat=newData @ n_eigVect                 #低维特征空间的数据
    reconMat=(lowDDataMat @ n_eigVect.T)+meanVal    #重构数据
    return lowDDataMat,reconMat
    
    
    
    
    
    
    
    




