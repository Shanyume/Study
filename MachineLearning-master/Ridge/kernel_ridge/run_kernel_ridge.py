#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel Ridge 回归测试脚本
============================

用 Iris 数据集对比手写 KRR 与 sklearn 实现。

Kernel Ridge Regression（核岭回归）原理：
    1. 用核函数 K(x_i, x_j) 计算样本间的相似度矩阵 K（Gram 矩阵）
    2. 求解 α = (K + λI)^{-1} y，其中 λ = 1/C 为正则化强度
    3. 预测：y_hat = K(x_test, X_train) @ α

    与线性岭回归的区别：核岭回归通过核技巧可以学习非线性关系，
    而线性岭回归只能拟合线性函数。

对比实验：
    - 手写 KRR（高斯核）vs sklearn 线性回归 vs sklearn KRR
    - 评估指标：均方误差（MSE）

数据集：Iris（取部分特征和标签）
依赖：numpy、scikit-learn
"""
import csv, os, sys
import numpy as np
from kernel_ridge import KernelRidge
filepath = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录，用于拼接数据文件路径

def readData(filename, header=True):
    """
    读取 CSV 数据文件

    :param filename: 文件路径
    :param header: 是否有表头行
    :return: (数据矩阵, 表头)，数据矩阵为二维 numpy 数组
    """
    data, header = [], None
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')  # 逗号分隔读取
        if header:
            header = next(spamreader)  # 有表头时先读一行
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))  # 数据转成二维数组

def calc_mse(y, y_hat):
    """
    计算均方误差（Mean Squared Error）

    MSE = mean((y - y_hat)^2)
    衡量预测值与真实值的平均偏差平方，越小越好

    :param y: 真实值数组
    :param y_hat: 预测值数组
    :return: 标量 MSE 值
    """
    return np.nanmean(((y - y_hat) ** 2))

def test_main(filename='small_data/iris-virginica.txt', C=1.0, kernel_type='linear'):
    """
    Kernel Ridge 回归完整测试流程

    对比三个模型：
        1. 手写 KRR（高斯核，C=0.1，gamma=5.0）
        2. sklearn 线性回归（基线）
        3. sklearn KRR（RBF 核，gamma=5，alpha=10）

    :param filename: 数据文件路径（相对于脚本目录）
    :param C: 正则化参数（此处未使用，模型内部固定为 0.1）
    :param kernel_type: 核函数类型（此处未使用，模型内部固定为高斯核）
    """
    # ---- 加载数据 ----
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)  # 读取 CSV（无表头）
    data = data.astype(float)  # 字符串转浮点

    # ---- 划分特征和标签 ----
    X, y = data[:,0:-1], data[:,-1].astype(int)  # 前几列为特征，最后一列为标签
    y = y[np.newaxis,:]  # 转成 (1, n) 行向量，便于矩阵运算
    print(X.shape)
    print(y.shape)

    # ---- 模型 1：手写核岭回归（高斯核）----
    # 高斯核（RBF）：K(x_i, x_j) = exp(-γ ||x_i - x_j||^2)
    # C=0.1 为正则化强度的倒数（λ = 1/C = 10），gamma=5.0 控制核宽度
    model = KernelRidge(kernel_type='gaussian', C=0.1, gamma=5.0)
    model.fit(X, y)
    y_hat = model.predict(x_train=X, x_test=X)  # 在训练集上预测（演示用）
    mse = calc_mse(y, y_hat)
    print(("mse of KRR:\t%.3f" % (mse)))

    # ---- 模型 2：sklearn 线性回归（基线）----
    # 线性回归只能拟合线性函数，作为对比基线
    from sklearn import linear_model
    ls = linear_model.LinearRegression()
    ls.fit(X, y[0,:])  # sklearn 要求一维目标值，取 y 的第一行
    y_ls = ls.predict(X)
    mse = calc_mse(y, y_ls)
    print(("mse of LS (from sklearn):\t%.3f" % (mse)))

    # ---- 模型 3：sklearn 核岭回归（RBF 核）----
    # sklearn 的 KernelRidge 用 alpha 表示正则化强度（α = K + alpha*I）
    # 对应我们的 C = 1/alpha，即 alpha=10 对应 C=0.1
    from sklearn.kernel_ridge import KernelRidge as KR2
    kr2 = KR2(kernel='rbf', gamma=5, alpha=10)
    kr2.fit(X, y[0, :])
    y_krr = kr2.predict(X)
    mse = calc_mse(y, y_krr)
    print(("mse of KRR (from sklearn):\t%.3f" % (mse)))


if __name__ == '__main__':
    # 用 iris-slwc.txt 数据运行对比测试
    test_main(filename='./small_data/iris-slwc.txt')
