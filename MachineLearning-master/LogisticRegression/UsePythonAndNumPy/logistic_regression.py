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
    """
    sigmoid 函数：把线性得分映射到 (0, 1)。
    含义：输出可解释为样本属于正类的概率 P(y=1|x) = 1/(1+e^(-z))，
    其中 z = x·w 是线性得分；z=0 时输出 0.5，
    z 的绝对值越大，输出越逼近 0 或 1。
    关键性质：sigmoid(z) 对 z 的导数 = sigmoid(z)*(1 - sigmoid(z))，
    正是下面梯度推导的核心，使更新公式简洁为 X^T (y - h)。
    数值注意：exp 溢出只发生在 inX 为大负数时，此时结果趋于 0；
    本数据集像素值为 0/1，z 通常不太大，未做 logit 稳定化处理。
    """
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataArray, labelArray, alpha, maxCycles):
    """
    全批量梯度上升优化权重。

    :param dataArray: 特征矩阵 (m, 1024)
    :param labelArray: 标签矩阵 (m, 1)
    :param alpha: 学习率
    :param maxCycles: 最大迭代次数
    :return: 权重矩阵 (1024, 1)

    推导（本算法的核心）：
        设 h_i = sigmoid(x_i·w) = P(y_i=1 | x_i, w)，
        全体训练样本的对数似然（等价于负交叉熵损失）：
            L(w) = Σ_i [ y_i·ln h_i + (1 - y_i)·ln(1 - h_i) ]
        对 w 求偏导：
            ∂L/∂w = Σ_i x_i·(y_i - h_i) = X^T (y - h)
        （利用了 h 关于 z 的导数 = h(1-h) 这一性质）
        梯度上升（最大化似然）：w := w + alpha · X^T (y - h)

    关于收敛：
        - L(w) 是凹函数（负交叉熵为凸），故存在全局最优，
          不存在局部最优陷阱；
        - 学习率 alpha 不宜过大：负交叉熵的 Hessian 为
          X^T diag(h(1-h)) X / m，要求 alpha 小于其最大特征值
          的 2 倍才能保证稳定下降（上升）；
        - 本实现为全批量（每轮都用全部样本），迭代次数固定
          为 maxCycles，未实现"梯度趋近 0 即提前停止"的
          收敛判据，因此 maxCycles 需给足；
        - 本数据集特征全是 0/1 像素（尺度已一致），
          无需额外归一化。
    """
    dataMat = np.asmatrix(dataArray)      # (m, n)
    labelMat = np.asmatrix(labelArray)    # (m, 1)
    m, n = np.shape(dataMat)
    # 说明：模型为 P(y=1|x) = sigmoid(x·w)，没有单独的偏置项 b。
    # 若需要偏置，可在特征中拼接一个恒为 1 的分量，
    # 效果与线性回归中"拼一列 1"的做法相同。
    # 权重初始化为全 1
    weigh = np.ones((n, 1))
    for i in range(maxCycles):
        # 前向：计算 sigmoid(Xw)
        h = sigmoid(dataMat * weigh)
        # 误差 = 真实标签 - 预测
        error = labelMat - h
        # 梯度上升更新权重
        weigh = weigh + alpha * dataMat.transpose() * error
        # 误差向量 (y - h) 的几何含义：
        #   预测对 -> |y - h| 小，该样本对权重拉动小；
        #   预测错 -> |y - h| 大，且方向指向修正方向，
        #   因此错得越狠的样本对权重的梯度贡献越大（类似"置信加权"）。
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
        # 阈值 0.5 的含义：sigmoid(z) > 0.5 当且仅当 z = x·w > 0，
        # 因此等价于对线性得分做符号函数判定 sign(x·w)，
        # 即判断样本位于决策边界 x·w = 0 的哪一侧。
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
    :param alpha: 学习率（步长；过大会导致对数似然振荡不收敛，
                  需小于 Hessian 最大特征值的 2 倍）
    :param maxCycles: 迭代次数（全批量梯度上升，本数据集 10 轮
                      已接近收敛；增加轮数可进一步降低错误率）
    """
    data, label = loadData(trainDir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    classfy(testDir, weigh)


if __name__ == "__main__":
    # 从脚本所在目录运行
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    digitRecognition(os.path.join(base, "train"), os.path.join(base, "test"))
