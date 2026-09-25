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
# 累计方差贡献率 = 前 n 个特征值之和 / 全部特征值之和
# （协方差矩阵的特征值即各主成分上的方差，故特征值占比 = 方差占比）
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
            #累计贡献率首次达到 percent（如 0.99）时，保留前 num 个主成分
            return num
# 数据中心化：每个特征减去均值，使均值为 0
def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal

def pca(dataMat, percent=0.99):
    #按列标准化前先去均值，保证主成分是相对数据中心的方向
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    
        
    eigVals,eigVects=np.linalg.eig(np.asmatrix(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    #covMat 是对称半正定矩阵，其特征值均 >= 0，每个特征值即对应主成分方向上的方差
    n=percent2n(eigVals,percent)                  #要达到percent的方差百分比，需要前n个特征向量
    eigValIndice=np.argsort(eigVals)              #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]     #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]          #最大的n个特征值对应的特征向量
    lowDDataMat=newData @ n_eigVect                 #低维特征空间的数据
    #投影：x_new = x_centered · W，W 的列向量是前 n 个主成分（特征向量）
    reconMat=(lowDDataMat @ n_eigVect.T)+meanVal    #重构数据
    #重构利用特征向量的正交性：x ≈ (x_centered · W) · W^T，再还原加回均值
    return lowDDataMat,reconMat
    
    
    
    
    
    
    
    




