#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bagging（自助聚合）独立实现
============================

Bagging 通过对训练集做有放回抽样（bootstrap）训练多个基学习器，
最后投票或平均，降低方差、提高泛化能力。Random Forest 是其经典扩展。

本脚本用 CART 决策树作为基学习器，Iris 数据集演示。

依赖：numpy、scikit-learn（仅用于数据加载和评估）
运行：python bagging.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest.random_forest import DecisionTree  # 复用已有 CART 决策树


class BaggingClassifier:
    """自助采样训练多个决策树，最后投票。"""

    def __init__(self, n_estimators=10, max_depth=4, random_state=42):
        """
        :param n_estimators: 基学习器数量
        :param max_depth: 每棵决策树的最大深度
        :param random_state: 随机种子，保证可复现
        """
        self.n = n_estimators
        self.max_depth = max_depth
        self.seed = random_state

    def fit(self, X, y):
        """训练：对每个基学习器做自助采样后训练一棵 CART 树。"""
        rng = np.random.RandomState(self.seed)
        self.models = []
        for i in range(self.n):
            # 有放回抽样：采样数量与原数据集相同
            idx = rng.randint(0, len(X), len(X))
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx])
            self.models.append(tree)
        return self

    def predict(self, X):
        """预测：每个基学习器先预测，再逐样本投票取多数。"""
        # 每行是一个基学习器对 X 的预测
        preds = np.array([m.predict(X) for m in self.models])
        # 对每个样本在基学习器之间投票
        return np.array([np.bincount(p.astype(int)).argmax() for p in preds.T])  # # 对每个样本在基学习器之间投票取多数


if __name__ == "__main__":
    # 加载 Iris 数据集
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # 训练 Bagging 模型
    model = BaggingClassifier(n_estimators=15).fit(Xtr, ytr)
    print("bagging test acc:", accuracy_score(yte, model.predict(Xte)))
