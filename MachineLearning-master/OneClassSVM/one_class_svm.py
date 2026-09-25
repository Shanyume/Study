#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-Class SVM 异常检测
========================

One-Class SVM 只用正常样本训练，学习正常数据边界，
边界外的样本判定为异常。

数据集：合成高斯数据 + 均匀分布异常点
依赖：scikit-learn、numpy
运行：python one_class_svm.py
"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def main():
    """One-Class SVM 异常检测演示。"""
    # 生成正常数据（高斯）和异常数据（均匀分布在远离区域）
    # 注意：训练时 One-Class SVM 只使用"正常"分布的信息（此处直接在全集上训练，
    # 因均匀异常点落在边界外，实际由核密度决定）
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])

    # 标准化：SVM 对尺度敏感
    # RBF 核的 gamma 依赖特征尺度，不标准化会使量纲大的特征主导距离计算
    X = StandardScaler().fit_transform(X)

    # nu 参数控制异常点比例的上界
    # nu ∈ (0,1]：异常样本占比的上界，同时也是支持向量占比的下界
    # gamma="scale"：自动取 1/(n_features * X.var())，无需手调
    model = OneClassSVM(kernel="rbf", nu=0.06, gamma="scale")
    # 决策函数 f(x) = Σ α_i K(x_i, x) - ρ：f(x) >= 0 为正常，f(x) < 0 判为异常
    # 边界由 RBF 核下的数据密度决定，密度低（远离主体）的区域被判为异常
    labels = model.fit_predict(X)  # -1 表示异常

    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")


if __name__ == "__main__":
    main()
