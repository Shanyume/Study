#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Softmax 回归（多分类逻辑回归）
================================

Softmax 回归是逻辑回归在多分类上的推广：

模型：
    P(y = k | x) = softmax(z)_k = exp(z_k) / Σ_j exp(z_j)
    其中 z_k = w_k · x + b_k

损失函数（交叉熵 + L2 正则）：
    L = -(1/m) Σ_i Σ_k y_ik * log(P(y=k|x_i)) + λ/(2m) ||W||²

梯度：
    ∂L/∂W = (1/m) X^T (P - Y) + λ W
    ∂L/∂b = (1/m) Σ_i (P_i - Y_i)

其中 Y 是 one-hot 编码，P 是 softmax 输出概率。

数据集：sklearn Iris（3 分类）
依赖：numpy、scikit-learn
运行：python softmax_regression.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class SoftmaxRegression:
    """NumPy 实现 Softmax 多分类回归。"""

    def __init__(self, lr=0.1, n_iters=500, l2=1e-4):
        """
        :param lr: 学习率
        :param n_iters: 最大迭代次数
        :param l2: L2 正则强度
        """
        self.lr = lr; self.n_iters = n_iters; self.l2 = l2

    def _softmax(self, z):
        """数值稳定的 softmax：减去每行最大值防止 exp 溢出。"""
        z = z - z.max(axis=1, keepdims=True)  # # 减去每行最大值，防止 exp 溢出
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)  # # softmax：指数归一化为概率分布

    def fit(self, X, y):
        """全批量梯度下降训练 softmax 回归。"""
        X = np.asarray(X); y = np.asarray(y)
        n, d = X.shape
        self.k = len(np.unique(y))  # 类别数
        # one-hot 编码标签
        Y = np.eye(self.k)[y]
        # 初始化权重和偏置
        self.W = np.zeros((d, self.k))
        self.b = np.zeros(self.k)
        for _ in range(self.n_iters):
            # 前向：计算概率 P(y=k|x)
            P = self._softmax(X @ self.W + self.b)
            # 梯度：交叉熵对 W 和 b 的偏导 + L2 正则
            grad_w = X.T @ (P - Y) / n + self.l2 * self.W  # # 交叉熵梯度 + L2 正则
            grad_b = (P - Y).mean(axis=0)  # # 偏置梯度
            # 更新参数
            self.W -= self.lr * grad_w  # # 沿负梯度方向更新权重
            self.b -= self.lr * grad_b  # # 更新偏置
        return self

    def predict_proba(self, X):
        """返回每个类别的概率。"""
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        """返回概率最大的类别。"""
        return self.predict_proba(X).argmax(axis=1)  # # 返回概率最大的类别


if __name__ == "__main__":
    # 加载 Iris 数据集
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # 标准化特征
    scaler = StandardScaler().fit(Xtr)
    model = SoftmaxRegression(lr=0.2, n_iters=1000).fit(scaler.transform(Xtr), ytr)
    print("Softmax test acc:", accuracy_score(yte, model.predict(scaler.transform(Xte))))
