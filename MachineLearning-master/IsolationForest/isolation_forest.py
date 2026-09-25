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
    # 正常点集中在原点附近（各维均值 0、标准差 1），异常点均匀散布在 [-8, 8]^2
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])

    # contamination 参数告诉模型预期的异常比例
    # 用于设定异常分数阈值：将分数最高的约 6% 的样本判为异常（本例 320 样本约 20 个）
    # n_estimators=100 即森林中 iTree 的数量，路径长度取均值以降噪
    model = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
    # fit_predict：在 X 上训练（无监督，不使用标签）并预测；
    # 异常分数基于平均隔离路径长度：路径越短分数越高（越异常）
    labels = model.fit_predict(X)  # -1 表示异常

    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")


if __name__ == "__main__":
    main()
