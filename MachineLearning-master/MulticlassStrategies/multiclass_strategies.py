#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多分类策略：One-vs-All 与 One-vs-One
======================================

sklearn 的二分类器默认通过多分类策略扩展到多分类：
    - One-vs-Rest (OvA)：每个类别训练一个二分类器
    - One-vs-One (OvO)：每对类别训练一个二分类器

数据集：sklearn Iris（3 分类）
依赖：scikit-learn
运行：python multiclass_strategies.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


def main():
    """OvA 与 OvO 对比。"""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # One-vs-All：k 个类别训练 k 个二分类器
    # 每个二分类器学习"类别 c vs 其余所有类别"（二分类标签为 {c} vs {非c}），
    # 预测时对每个分类器得分取最大者（等价于取概率最大类）
    # max_iter=1000 提高 LBFGS 最大迭代次数，避免逻辑回归不收敛警告
    ova = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    ova.fit(Xtr, ytr)
    print("OvA test acc:", ova.score(Xte, yte))

    # One-vs-One：k*(k-1)/2 个二分类器
    # 每对类别 (c_i, c_j) 只训练一个"两者二分类"，预测时多数表决
    # OvO 训练规模小（每个子问题只用两类样本）、预测时需 O(k^2) 次分类
    ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    ovo.fit(Xtr, ytr)
    print("OvO test acc:", ovo.score(Xte, yte))


if __name__ == "__main__":
    main()
