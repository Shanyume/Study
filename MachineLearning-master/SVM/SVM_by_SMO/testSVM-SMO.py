#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM SMO 测试脚本
==================

用 Iris 二分类数据测试 SMO SVM 实现：
    - 读取 CSV 数据（无表头）
    - 训练 SVCSMO 模型
    - 输出支持向量数、偏置、权重、准确率、MSE、迭代次数

数据集：SVM/SVM_by_SMO/small_data/iris-slwc.txt
依赖：numpy
"""

import csv, os, sys
import numpy as np
from SVCSMO import SVCSMO
filepath = os.path.dirname(os.path.abspath(__file__))  # # 当前脚本所在目录，用于拼接数据文件路径

def readData(filename, header=True):
    """读取 CSV 文件，返回 (数据矩阵, 表头)。"""
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')  # # 逗号分隔读取
        if header:
            header = next(spamreader)  # # 有表头时先读一行表头
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))  # # 数据转为二维数组

def calc_acc(y, y_hat):
    """计算准确率：分别统计预测为 1 和 -1 时的正确数。"""
    idx = np.where(y_hat == 1)  # # 找到预测为 1 的样本索引
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)  # # 找到预测为 -1 的样本索引
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

def calc_mse(y, y_hat):
    """计算均方误差。"""
    return np.nanmean(((y - y_hat) ** 2))

def test_main(filename='data/iris-virginica.txt', C=1.0, kernel_type='linear', epsilon=0.001):
    """SVM SMO 完整测试流程：加载数据 -> 训练 -> 预测 -> 输出指标。"""
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)  # # 读取 CSV（无表头），路径拼脚本目录
    data = data.astype(float)  # # 字符串转浮点

    # 划分特征和标签
    X, y = data[:,0:-1], data[:,-1].astype(int)

    # 初始化 SMO SVM 模型
    model = SVCSMO()  # # 使用默认参数：线性核、C=1.0

    # 训练模型，返回支持向量数组和迭代次数
    support_vectors, iterations = model.fit(X, y)

    # 支持向量数量
    sv_count = support_vectors.shape[0]

    # 用训练好的模型预测
    y_hat = model.predict(X)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)  # # 准确率 = 正确预测数 / 样本总数
    mse = calc_mse(y, y_hat)  # # 均方误差

    # # 输出模型参数与评估指标
    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("mse:\t%.3f" % (mse))
    print("Converged after %d iterations" % (iterations))

if __name__ == '__main__':
    param = {}  # # 测试参数
    param['filename'] = './small_data/iris-slwc.txt'
    param['C'] = 0.1  # # 较小的 C：软间隔容忍部分误分
    param['kernel_type'] = 'linear'
    param['epsilon'] = 0.001


    test_main(**param)  # # 运行完整测试流程

