#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM QP 测试脚本
=================

用合成二维数据测试 QP SVM 的三种场景：
- test_linear：线性可分
- test_non_linear：非线性可分（高斯核）
- test_soft：软间隔（重叠数据）

依赖：numpy、cvxopt、matplotlib
"""
from SVCQP import *
import pylab as pl

def gen_lin_separable_data():
    """生成线性可分的二分类数据：两类分别服从不同均值的高斯分布。"""
    # 类别 1 的均值和协方差
    mean1 = np.array([0, 2])  # # 类别 +1 的均值（左上方）
    mean2 = np.array([2, 0])  # # 类别 -1 的均值（右下方）
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])  # # 相关协方差：两维强相关，样本沿对角线方向拉伸
    # 从多元高斯分布采样 100 个样本
    X1 = np.random.multivariate_normal(mean1, cov, 100)  # # 类 +1：均值 mean1，共 100 个样本
    y1 = np.ones(len(X1))           # 标签 +1
    X2 = np.random.multivariate_normal(mean2, cov, 100)  # # 类 -1：均值 mean2，共 100 个样本
    y2 = np.ones(len(X2)) * -1      # 标签 -1
    return X1, y1, X2, y2


def gen_non_lin_separable_data():
    """生成非线性可分数据：每类由两个高斯簇组成，形成月牙形/环形结构。"""
    mean1 = [-1, 2]   # # 类 +1 簇 1（左上）
    mean2 = [1, -1]   # # 类 -1 簇 1（右中）
    mean3 = [4, -4]   # # 类 +1 簇 2（右下）
    mean4 = [-4, 4]   # # 类 -1 簇 2（左中）
    cov = [[1.0,0.8], [0.8, 1.0]]  # # 相关协方差，使两维样本相关
    # 类别 1 由两个簇拼接（左上和右下）
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))  # # 两个簇纵向拼接
    y1 = np.ones(len(X1))
    # 类别 -1 由两个簇拼接（右上和左下）
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))  # # 两个簇纵向拼接
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_lin_separable_overlap_data():
    """生成线性可分但有重叠的数据：两类协方差更大，有部分重叠。"""
    mean1 = np.array([0, 2])  # # 类 +1 的均值
    mean2 = np.array([2, 0])  # # 类 -1 的均值
    # 更大的协方差 -> 数据更分散、有重叠
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])  # # 比 gen_lin_separable_data 更大的协方差
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2):
    """把两类数据前 90 个样本合并为训练集。"""
    X1_train = X1[:90]     # # 每类取前 90 个
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    # 纵向拼接特征，横向拼接标签
    X_train = np.vstack((X1_train, X2_train))  # # 特征：两类纵向拼成 (180, 2)
    y_train = np.hstack((y1_train, y2_train))  # # 标签：两类横向拼成 (180,)
    return X_train, y_train


def split_test(X1, y1, X2, y2):
    """把两类数据第 90 个之后作为测试集。"""
    X1_test = X1[90:]      # # 每类取第 90 个之后
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))  # # 特征：两类纵向拼成 (20, 2)
    y_test = np.hstack((y1_test, y2_test))  # # 标签：两类横向拼成 (20,)
    return X_test, y_test

def plot_margin(X1_train, X2_train, clf):
    """绘制线性 SVM 的决策边界和间隔边界（w·x+b=±1）。"""
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        # # 由 w·x + b = c 解出 y = (-w[0]*x - b + c) / w[1]
        return (-w[0] * x - b + c) / w[1]

    pl.plot(X1_train[:,0], X1_train[:,1], "ro")        # # 红点：类 +1
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")        # # 蓝点：类 -1
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g") # # 绿点：支持向量

    # w.x + b = 0
    # # 决策边界（w·x + b = 0），取 x = ±4 两个端点连成黑实线
    a0 = -4; a1 = f(a0, clf.w, clf.b)
    b0 = 4; b1 = f(b0, clf.w, clf.b)
    pl.plot([a0,b0], [a1,b1], "k")

    # w.x + b = 1
    # # 间隔边界之一（w·x + b = +1），黑虚线
    a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
    pl.plot([a0,b0], [a1,b1], "k--")

    # w.x + b = -1
    # # 间隔边界之二（w·x + b = -1），黑虚线；两虚线之间即为间隔 2/||w||
    a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
    b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
    pl.plot([a0,b0], [a1,b1], "k--")

    pl.axis("tight")
    pl.show()

def plot_contour(X1_train, X2_train, clf):
    """绘制非线性 SVM 的决策边界（等高线 w·x+b=0）。"""
    pl.plot(X1_train[:,0], X1_train[:,1], "ro")        # # 红点：类 +1
    pl.plot(X2_train[:,0], X2_train[:,1], "bo")        # # 蓝点：类 -1
    pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g") # # 绿点：支持向量

    # # 在 [-6, 6] 范围生成 50x50 网格，用于计算决策函数值并画等高线
    X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])  # # 把网格展平成二维样本矩阵，用于预测决策边界
    Z = clf.project(X).reshape(X1.shape)  # # 网格上每个点的决策函数值 f(x)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')                # # f(x) = 0 的黑色等高线：决策边界
    pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')         # # f(x) = -1 的灰色等高线：一侧间隔边界
    pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')         # # f(x) = +1 的灰色等高线：另一侧间隔边界

    pl.axis("tight")
    pl.show()

def test_linear():
    """测试线性可分场景：用线性核 SVM，绘制决策边界。"""
    X1, y1, X2, y2 = gen_lin_separable_data()        # # 线性可分数据
    X_train, y_train = split_train(X1, y1, X2, y2)   # # 前 90 个样本作训练集
    X_test, y_test = split_test(X1, y1, X2, y2)      # # 其余作测试集

    clf = SVM()  # 默认线性核，硬间隔
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)  # # 预测正确的样本数
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_margin(X_train[y_train==1], X_train[y_train==-1], clf)  # # 绘图：决策边界 + 间隔边界

def test_non_linear():
    """测试非线性可分场景：用高斯核 SVM，绘制等高线决策边界。"""
    X1, y1, X2, y2 = gen_non_lin_separable_data()  # # 非线性可分数据
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)

    # X_train = np.load('inputClf/X_train.npy')
    # y_train = np.load('inputClf/y_train.npy')
    # X_test = np.load('inputClf/X_test.npy')
    # y_test = np.load('inputClf/y_test.npy')
    clf = SVM(gaussian_kernel, C=1)  # 高斯核 + 软间隔
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)  # # 预测正确的样本数
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)  # # 绘图：等高线决策边界

def test_soft():
    """测试软间隔场景：数据有重叠，用较小的 C 允许部分错误。"""
    X1, y1, X2, y2 = gen_lin_separable_overlap_data()  # # 有重叠的线性可分数据
    X_train, y_train = split_train(X1, y1, X2, y2)
    X_test, y_test = split_test(X1, y1, X2, y2)



    clf = SVM(C=0.1)  # 较小的 C：更容忍错误，边界更平滑
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    correct = np.sum(y_predict == y_test)  # # 预测正确的样本数
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)  # # 绘图：等高线决策边界


if __name__ == "__main__":
    # # 三种场景：test_linear / test_non_linear / test_soft，按需取消注释
    test_non_linear()
    #test_soft()