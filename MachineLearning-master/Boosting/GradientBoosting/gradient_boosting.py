#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Boosting 简化实现
============================

梯度提升（Gradient Boosting）的核心思想：

    1. 初始化模型 F_0 为常数（通常为 0）
    2. 每轮迭代 m = 1..M：
        a. 计算当前模型 F_{m-1} 的负梯度（伪残差）
           对 logistic loss：r_i = y_i / (1 + exp(y_i * F(x_i)))
        b. 用一个基学习器（决策树桩）拟合残差 r
        c. 更新模型：F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    3. 最终模型是所有树的加权和

对 logistic loss，负梯度等价于"当前模型预测错误的样本权重更大"，
即每轮更关注上一轮分错的样本。

本脚本用决策树桩（深度 1 的回归树）作为基学习器，演示二分类。

数据集：sklearn make_moons（二分类）
依赖：numpy、scikit-learn
运行：python gradient_boosting.py
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SimpleTreeStump:
    """深度为 1 的回归树桩，用于拟合残差。"""

    def fit(self, X, y):
        """寻找最优分裂特征和阈值，按叶节点均值预测。"""
        # 这里 y 传入的是伪残差（负梯度值），而非真实标签
        # 因此树桩按回归方式训练：叶节点输出残差均值
        best = (None, None, np.inf)
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                # MSE 损失：左右子树的残差平方和
                err = ((left - left.mean())**2).sum() + ((right - right.mean())**2).sum()
                if err < best[2]:
                    best = (j, t, err)
        self.j, self.t = best[0], best[1]
        # 叶节点均值作为预测值
        self.left = y[X[:, self.j] <= self.t].mean() if self.j is not None else 0.0
        self.right = y[X[:, self.j] > self.t].mean() if self.j is not None else 0.0
        return self

    def predict(self, X):
        """根据阈值分裂预测。"""
        # x_j <= t 返回左叶均值，否则返回右叶均值
        return np.where(X[:, self.j] <= self.t, self.left, self.right)


class GradientBoostingClassifier:
    """梯度提升分类器（logistic loss）。"""

    def __init__(self, n_estimators=50, learning_rate=0.1):
        """
        :param n_estimators: 迭代轮数（树的数量）
        :param learning_rate: 学习率（步长收缩）
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.trees = []

    def fit(self, X, y):
        """逐轮拟合负梯度。"""
        # 标签转为 {-1, 1}
        # 用 ±1 标签 + logistic loss 可避免 softmax 的参数冗余，且与负梯度推导一致
        y = np.where(y == 1, 1.0, -1.0)
        # 初始化 F_0(x) = 0（logistic loss 下最优常数解为 0）
        F = np.zeros(len(y))  # 当前模型的决策函数值
        for _ in range(self.n_estimators):
            # logistic loss 的负梯度：y / (1 + exp(y*F))
            # 即当前模型对样本 i 的"加权错误率"：F 越错，伪残差越大（上界为 1）
            residual = y / (1 + np.exp(y * F))  # # logistic loss 的负梯度（伪残差）
            # 用树桩拟合残差
            # h_m 拟合的是伪残差而非真实标签（区别于 AdaBoost 直接拟合标签）
            tree = SimpleTreeStump().fit(X, residual)
            # 更新模型：F += lr * tree
            # F_m(x) = F_{m-1}(x) + lr * h_m(x)，lr 是收缩系数（shrinkage），
            # 防止单棵树主导并抑制过拟合，lr 越小通常需要越多的树
            F += self.lr * tree.predict(X)  # # 用学习率收缩后更新模型
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        """累加所有树的加权预测得到决策函数值。"""
        # F(x) = Σ_m lr * h_m(x)，即所有树桩贡献的加权和
        return sum(self.lr * t.predict(X) for t in self.trees)  # # 累加所有树的加权预测

    def predict(self, X):
        """决策函数值 > 0 预测为 1，否则为 0。"""
        # 对应概率 P(y=1|x) = 1/(1+exp(-F(x)))，取 0.5 阈值等价于判断 F(x) 的正负
        return (self.decision_function(X) > 0).astype(int)  # # 决策函数值 > 0 预测为 1


if __name__ == "__main__":
    # 生成双月牙形二分类数据
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2)
    model.fit(Xtr, ytr)
    print("train acc:", accuracy_score(ytr, model.predict(Xtr)))
    print("test  acc:", accuracy_score(yte, model.predict(Xte)))
