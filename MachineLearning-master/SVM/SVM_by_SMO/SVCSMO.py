# -*- coding: utf-8 -*-
"""
SVM by SMO（序列最小优化 SVM）
================================

SMO（Sequential Minimal Optimization）是 SVM 的经典训练算法：

核心思想：
    - 每次迭代只优化两个 alpha 变量（i 和 j），其他保持不变
    - 因为等式约束 Σ alpha_i y_i = 0，所以必须同时更新两个 alpha
    - 逐轮遍历所有样本，直到 alpha 变化很小（收敛）或达到最大迭代次数

SMO 更新公式：
    alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / k_ij
    其中 k_ij = K(x_i,x_i) + K(x_j,x_j) - 2*K(x_i,x_j)
    E_k = h(x_k) - y_k 是预测误差

支持核函数：
    - linear：线性核
    - quadratic：二次核
    - gaussian：高斯核（RBF）

依赖：numpy
"""
from __future__ import division, print_function
from numpy import linalg
import os
import numpy as np
import random as rnd
filepath = os.path.dirname(os.path.abspath(__file__))  # # 当前脚本所在目录（用于定位数据文件）

class SVCSMO():
    """
    使用 SMO 算法训练的支持向量分类器（简化实现）。

    参数：
        max_iter: 最大迭代轮数
        kernel_type: 核类型，'linear' / 'quadratic' / 'gaussian'
        C: 正则化参数（软间隔的惩罚系数）
        epsilon: 收敛阈值，alpha 变化小于该值则停止
        sigma: 高斯核的 σ 参数

    属性：
        w: 线性核时的权重向量
        b: 偏置项
    """
    def __init__(self, max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.001, sigma=5.0):
        """
        :param max_iter: maximum iteration
        :param kernel_type: Kernel type to use in training.
                        'linear' use linear kernel function.
                        'quadratic' use quadratic kernel function.
                        'gaussian' use gaussian kernel function
        :param C: Value of regularization parameter C
        :param epsilon: Convergence value.
        :param sigma: parameter for gaussian kernel
        """
        # 核函数注册表：字符串类型 -> 对应的核函数方法
        self.kernels = {
            'linear' : self.kernel_linear,
            'quadratic' : self.kernel_quadratic,
            'gaussian' : self.kernel_gaussian
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma
    def fit(self, X, y):
        """
        SMO 训练：迭代优化 alpha 变量。

        :param X: 训练特征矩阵 (n_samples, n_features)
        :param y: 训练标签 (+1 / -1)
        :return: (支持向量数组, 迭代次数)
        """
        # Initialization
        n, d = X.shape[0], X.shape[1]  # # n: 样本数，d: 特征维度
        alpha = np.zeros((n))  # # 拉格朗日乘子 alpha 初始化为 0
        kernel = self.kernels[self.kernel_type]  # # 按 kernel_type 选择核函数
        count = 0  # # 迭代轮数计数
        while True:
            count += 1  # # 扫描完一轮所有样本记一次迭代
            alpha_prev = np.copy(alpha)  # # 保存本轮前的 alpha，用于收敛判断
            for j in range(0, n):  # # 逐一遍历样本 j（作为第二个被优化的变量）
                i = self.get_rnd_int(0, n-1, j)  # # 随机选一个 i != j 的样本 # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)  # # 计算 K(x_i,x_i) + K(x_j,x_j) - 2K(x_i,x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]  # # 记录更新前的 alpha_j、alpha_i
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)  # # 计算alpha[j]的可行域下界L和上界H

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)  # # 由当前 alpha 计算 w = Σ alpha_i * y_i * x_i
                self.b = self.calc_b(X, y, self.w)  # # 由当前 w 估计偏置 b

                # Compute E_i, E_j
                # # 预测误差 E(x) = h(x) - y，SMO 更新式需要 E_i 与 E_j 之差
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                # # 按 SMO 公式更新 alpha_j，裁剪到 [L, H] 后再由约束反推 alpha_i
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij  # # SMO 更新公式
                alpha[j] = max(alpha[j], L)  # # 裁剪到可行域下界
                alpha[j] = min(alpha[j], H)  # # 裁剪到可行域上界

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])  # # 保持等式约束 y_i*alpha_i + y_j*alpha_j 不变

            # Check convergence
            # # 本轮 alpha 的 L2 变化量小于 epsilon 即视为收敛
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:  # # 超过最大迭代轮数仍未收敛则停止
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        # Compute final model parameters
        # # 收敛后用最终 alpha 重算 b；线性核还需重算 w
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        # # 支持向量 = alpha > 0 的样本（包括 alpha = C 的边界支持向量）
        alpha_idx = np.where(alpha > 0)[0]  # # 支持向量：alpha > 0 的样本索引
        support_vectors = X[alpha_idx, :]
        return support_vectors, count
    def predict(self, X):
        """预测：返回 sign(w·x + b)。"""
        return self.h(X, self.w, self.b)
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)  # # 对每个样本计算 y_i - w·x_i
        return np.mean(b_tmp)  # # b 取均值（标准做法是对 0<alpha_i<C 的支持向量取均值，此处简化为全体样本）
    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)  # # w = Σ alpha_i * y_i * x_i（仅线性核可显式表示）
    # Prediction
    # # 决策函数 h(x) = sign(w·x + b)
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)
    # Prediction error
    # # 预测误差 E(x_k) = h(x_k) - y_k
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k
    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        # 由"y_i*alpha_i + y_j*alpha_j 保持不变"及 0 <= alpha <= C 推出 alpha_j 的可行范围 [L, H]
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))  # # 异类样本：L = max(0, alpha_j - alpha_i)，H = min(C, C - alpha_i + alpha_j)
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))  # # 同类样本：L = max(0, alpha_i + alpha_j - C)，H = min(C, alpha_i + alpha_j)
    def get_rnd_int(self, a,b,z):
        # 在 [a, b] 内随机取一个与 z 不同的整数，保证选出的 i != j
        i = z
        while i == z:
            i = rnd.randint(a,b)
        return i
    # Define kernels
    def kernel_linear(self, x1, x2):
        # 线性核：K(x1, x2) = x1 · x2
        return np.dot(x1, x2.T)
    def kernel_quadratic(self, x1, x2):
        # 二次核：K(x1, x2) = (x1 · x2)^2
        return (np.dot(x1, x2.T) ** 2)
    def kernel_gaussian(self, x1, x2, sigma=5.0):
        # 高斯核（RBF）：K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))
        if self.sigma:
            sigma = self.sigma
        return np.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))
