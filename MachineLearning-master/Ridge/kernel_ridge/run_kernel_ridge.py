#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel Ridge 回归测试脚本
============================

用 Iris 数据集对比手写 KRR 与 sklearn 实现。

依赖：numpy、scikit-learn
"""
import csv, os, sys
import numpy as np
from kernel_ridge import KernelRidge
filepath = os.path.dirname(os.path.abspath(__file__))  # # 当前脚本所在目录，用于拼接数据文件路径

# # 读取 CSV 数据文件，返回 (数据矩阵, 表头)
def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')  # # 逗号分隔读取
        if header:
            header = next(spamreader)  # # 有表头时先读一行
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))  # # 数据转成二维数组

# # 计算均方误差：MSE = mean((y - y_hat)^2)
def calc_mse(y, y_hat):
    return np.nanmean(((y - y_hat) ** 2))

def test_main(filename='small_data/iris-virginica.txt', C=1.0, kernel_type='linear'):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)  # # 读取 CSV（无表头）
    data = data.astype(float)  # # 字符串转浮点

    # Split data
    X, y = data[:,0:-1], data[:,-1].astype(int)  # # 前几列为特征，最后一列为标签
    y = y[np.newaxis,:]  # # 转成 (1, n) 行向量，便于矩阵相乘
    print(X.shape)
    print(y.shape)



    # fit our model
    # # 自实现核岭回归：高斯核 + 正则化 C=0.1 + gamma=5.0
    model = KernelRidge(kernel_type='gaussian', C=0.1, gamma=5.0)
    model.fit(X, y)
    y_hat = model.predict(x_train=X, x_test=X)  # # 在训练集上预测（演示用）
    mse = calc_mse(y, y_hat) # Calculate accuracy
    print(("mse of KRR:\t%.3f" % (mse)))

    # fit linear model for test
    # # 对比基线 1：sklearn 线性回归
    from sklearn import linear_model
    ls = linear_model.LinearRegression()
    ls.fit(X, y[0,:])  # # sklearn 要求一维目标值，取 y 的第一行
    y_ls = ls.predict(X)
    mse = calc_mse(y, y_ls)
    print(("mse of LS (from sklearn):\t%.3f" % (mse)))

    # fit KRR from sklearn for test
    # # 对比基线 2：sklearn 核岭回归（alpha 即正则化强度，对应 K + alpha*I）
    from sklearn.kernel_ridge import KernelRidge as KR2
    kr2 = KR2(kernel='rbf', gamma=5, alpha=10)
    kr2.fit(X, y[0, :])
    y_krr = kr2.predict(X)
    mse = calc_mse(y, y_krr)
    print(("mse of KRR (from sklearn):\t%.3f" % (mse)))





if __name__ == '__main__':
    # # 用 iris-slwc.txt 数据运行对比测试
    test_main(filename='./small_data/iris-slwc.txt')
