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
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])

    # 标准化：SVM 对尺度敏感
    X = StandardScaler().fit_transform(X)

    # nu 参数控制异常点比例的上界
    model = OneClassSVM(kernel="rbf", nu=0.06, gamma="scale")
    labels = model.fit_predict(X)  # -1 表示异常

    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")


if __name__ == "__main__":
    main()
