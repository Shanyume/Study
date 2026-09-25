#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest 异常检测
==========================

Isolation Forest 基于"异常点更容易被隔离"的思想：
    - 随机选特征和阈值递归划分
    - 异常点路径更短，更容易被隔离
    - 无监督，不需要标签

数据集：合成高斯数据 + 均匀分布异常点
依赖：scikit-learn、numpy
运行：python isolation_forest.py
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def main():
    """Isolation Forest 异常检测演示。"""
    # 正常数据（高斯）+ 异常数据（远离中心）
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])

    # contamination 参数告诉模型预期的异常比例
    model = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
    labels = model.fit_predict(X)  # -1 表示异常

    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")


if __name__ == "__main__":
    main()
