#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Boosting 简化实现
============================

梯度提升通过逐轮拟合当前模型的负梯度（对 logistic loss 就是残差），
逐步修正错误。XGBoost / LightGBM 都基于这个思想。

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
        y = np.where(y == 1, 1.0, -1.0)
        F = np.zeros(len(y))  # 当前模型的决策函数值
        for _ in range(self.n_estimators):
            # logistic loss 的负梯度：y / (1 + exp(y*F))
            residual = y / (1 + np.exp(y * F))
            # 用树桩拟合残差
            tree = SimpleTreeStump().fit(X, residual)
            # 更新模型：F += lr * tree
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        """累加所有树的加权预测得到决策函数值。"""
        return sum(self.lr * t.predict(X) for t in self.trees)

    def predict(self, X):
        """决策函数值 > 0 预测为 1，否则为 0。"""
        return (self.decision_function(X) > 0).astype(int)


if __name__ == "__main__":
    # 生成双月牙形二分类数据
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2)
    model.fit(Xtr, ytr)
    print("train acc:", accuracy_score(ytr, model.predict(Xtr)))
    print("test  acc:", accuracy_score(yte, model.predict(Xte)))
