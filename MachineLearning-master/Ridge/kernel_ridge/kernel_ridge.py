# -*- coding: utf-8 -*-
"""
Kernel Ridge Regression（核岭回归）
======================================

核岭回归把核方法（Kernel）和岭回归（Ridge Regression）结合：

核心公式：
    alpha = (K + C * I)^(-1) * y
    y_test = K(x_test, x_train) @ alpha

其中：
    K: 核矩阵（Gram matrix），K[i,j] = kernel(x_i, x_j)
    C: 正则化参数，越大越平滑
    I: 单位矩阵

核函数：
    - linear：K(x,y) = x · y^T
    - quadratic：K(x,y) = (x · y^T)^2
    - gaussian：K(x,y) = exp(-||x-y||² / (2σ²))

Doc: https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf

依赖：numpy、scipy
"""
from numpy.linalg import inv
import numpy as np
from scipy import linalg


class KernelRidge():
    """
    核岭回归分类器（简化实现），使用闭式解训练。

    参数：
        kernel_type: 核类型，'linear' / 'quadratic' / 'gaussian'
        C: 正则化参数，越大越平滑
        gamma: 高斯核的 σ 参数

    属性：
        alphas: 训练后得到的系数 (n_samples, 1)
        kernel: 当前选用的核函数
    """

    def __init__(self, kernel_type='linear', C=1.0, gamma=5.0):
        """
        :param kernel_type: Kernel type to use in training.
                        'linear' use linear kernel function.
                        'quadratic' use quadratic kernel function.
                        'gaussian' use gaussian kernel function
        :param C: Value of regularization parameter C
        :param gamma: parameter for gaussian kernel or Polynomial kernel
        """
        self.kernels = {  # # 核函数注册表：字符串类型 -> 对应的核函数方法
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'gaussian': self.kernel_gaussian
        }
        self.kernel_type = kernel_type
        self.kernel = self.kernels[self.kernel_type]  # # 按类型选定本次使用的核函数
        self.C = C
        self.gamma = gamma

    # 定义核函数
    # # 核函数把输入映射到更高维（或隐式）空间中的内积，
    # # 使回归可以在特征空间中进行而无需显式变换
    def kernel_linear(self, x1, x2):
        """线性核：K(x1, x2) = x1 · x2^T。"""
        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        """二次核：K(x1, x2) = (x1 · x2^T)^2。"""
        return (np.dot(x1, x2.T) ** 2)

    def kernel_gaussian(self, x1, x2, gamma=5.0):
        """高斯核：K(x1, x2) = exp(-||x1-x2||² / (2σ²))，σ = gamma。"""
        # # gamma 越大，核值衰减越快（模型更平滑、更易欠拟合）
        gamma = self.gamma
        return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (gamma ** 2)))

    def compute_kernel_matrix(self, X1, X2):
        """
        compute kernel matrix (gram matrix) give two input matrix
        """

        # sample size
        n1 = X1.shape[0]  # # X1 的样本数（核矩阵行数）
        n2 = X2.shape[0]  # # X2 的样本数（核矩阵列数）

        # Gram matrix
        K = np.zeros((n1, n2))  # # 核矩阵：K[i, j] = kernel(X1[i], X2[j])
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        return K


    def fit(self, X, y):
        """
        训练 KRR：计算核矩阵并求闭式解 alpha = (K + C*I)^(-1) * y。

        :param X: 训练特征矩阵 (n_samples, n_features)
        :param y: 训练目标值 (n_samples, 1)
        :return: alpha 系数
        """
        K = self.compute_kernel_matrix(X, X)  # # 训练集上的核矩阵

        # # 推导：min 1/2 alpha^T K alpha + C ||alpha||^2 - alpha^T y
        # #        令导数为 0 得 (K + C*I) alpha = y
        self.alphas = np.dot(inv(K + self.C * np.eye(np.shape(K)[0])),  # # 闭式解：alpha = (K + C*I)^(-1) * y
                        y)

        return self.alphas

    def predict(self, x_train, x_test):
        """
        预测：y_test = K(x_test, x_train) @ alpha。

        :param x_train: 训练特征矩阵
        :param x_test: 测试特征矩阵
        :return: 预测值
        """
        """

        :param x_train: DxNtr array of Ntr train data points
                        with D features
        :param x_test:  DxNte array of Nte test data points
                        with D features
        :return: y_test, D2xNte array
        """

        k = self.compute_kernel_matrix(x_test, x_train)  # # 核矩阵：K(x_test, x_train)，形状 (n_test, n_train)

        y_test = np.dot(k, self.alphas)  # # 预测：K(x_test, x_train) @ alpha
        # # 即 f(x) = Σ_j alpha_j * K(x, x_j)，由核表示定理在特征空间中线性组合训练样本
        return y_test

