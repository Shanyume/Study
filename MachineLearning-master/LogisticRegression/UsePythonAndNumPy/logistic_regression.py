#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression（逻辑回归）Python + NumPy 实现
=====================================================

逻辑回归是线性分类模型，通过 sigmoid 函数把线性得分映射到 [0,1]，
用梯度上升最大化对数似然。

本脚本实现二分类逻辑回归，应用于手写数字（0 和 1）识别：
    - loadData：读取 32×32 文本图像转为 1024 维向量
    - gradAscent：全批量梯度上升优化权重
    - classfy：用学到的权重预测测试集

数据集：二分类手写数字（train/ 和 test/ 目录下的 txt 文件）
依赖：numpy
运行：python logistic_regression.py
"""
import numpy as np
from os import listdir


def loadData(direction):
    """
    读取目录下所有 32×32 文本图像，转为特征矩阵和标签矩阵。

    :param direction: 数据目录路径
    :return: dataArray (m, 1024) 特征矩阵, labelArray (m, 1) 标签矩阵
    """
    # 列出目录下所有文件
    trainfileList = listdir(direction)
    m = len(trainfileList)
    # 每个文件是 32×32 = 1024 个像素
    dataArray = np.zeros((m, 1024))
    labelArray = np.zeros((m, 1))
    for i in range(m):
        # 每个文件转为 1×1024 向量
        returnArray = np.zeros((1, 1024))
        filename = trainfileList[i]
        with open(f"{direction}/{filename}") as fr:
            for j in range(32):
                lineStr = fr.readline()
                for k in range(32):
                    # 第 j 行第 k 列像素放到向量的第 32*j+k 位
                    returnArray[0, 32*j+k] = int(lineStr[k])
        dataArray[i, :] = returnArray  # 存储特征向量
        # 从文件名解析标签：如 "1_10.txt" -> 1
        filename0 = filename.split(".")[0]
        label = filename0.split("_")[0]
        labelArray[i] = int(label)
    return dataArray, labelArray


def sigmoid(inX):
    """sigmoid 函数：把线性得分映射到 (0, 1)。"""
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataArray, labelArray, alpha, maxCycles):
    """
    全批量梯度上升优化权重。

    :param dataArray: 特征矩阵 (m, 1024)
    :param labelArray: 标签矩阵 (m, 1)
    :param alpha: 学习率
    :param maxCycles: 最大迭代次数
    :return: 权重矩阵 (1024, 1)
    """
    dataMat = np.asmatrix(dataArray)      # (m, n)
    labelMat = np.asmatrix(labelArray)    # (m, 1)
    m, n = np.shape(dataMat)
    # 权重初始化为全 1
    weigh = np.ones((n, 1))
    for i in range(maxCycles):
        # 前向：计算 sigmoid(Xw)
        h = sigmoid(dataMat * weigh)
        # 误差 = 真实标签 - 预测
        error = labelMat - h
        # 梯度上升更新权重
        weigh = weigh + alpha * dataMat.transpose() * error
    return weigh


def classfy(testdir, weigh):
    """
    用学到的权重对测试集分类并输出错误率。

    :param testdir: 测试数据目录
    :param weigh: 权重矩阵
    """
    dataArray, labelArray = loadData(testdir)
    dataMat = np.asmatrix(dataArray)
    labelMat = np.asmatrix(labelArray)
    # 预测概率
    h = sigmoid(dataMat * weigh)
    m = len(h)
    error = 0.0
    for i in range(m):
        # 概率 > 0.5 预测为 1，否则为 0
        if h[i, 0] > 0.5:
            print(int(labelMat[i, 0]), "is classified as: 1")
            if labelMat[i, 0] != 1:
                error += 1
                print("error")
        else:
            print(int(labelMat[i, 0]), "is classified as: 0")
            if labelMat[i, 0] != 0:
                error += 1
                print("error")
    print("error rate is:", "%.4f" % (error/m))


def digitRecognition(trainDir, testDir, alpha=0.07, maxCycles=10):
    """
    完整流程：加载数据 -> 训练 -> 测试。

    :param trainDir: 训练数据目录
    :param testDir: 测试数据目录
    :param alpha: 学习率
    :param maxCycles: 迭代次数
    """
    data, label = loadData(trainDir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    classfy(testDir, weigh)


if __name__ == "__main__":
    # 从脚本所在目录运行
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    digitRecognition(os.path.join(base, "train"), os.path.join(base, "test"))
