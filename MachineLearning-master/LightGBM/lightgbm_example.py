#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM 分类示例
===================

LightGBM 是基于直方图的快速梯度提升实现，训练速度快，适合大数据集。

数据集：sklearn Iris（3 分类）
依赖：lightgbm、scikit-learn
运行：python lightgbm_example.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


def main():
    """训练 LightGBM 并输出准确率和特征重要性。"""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # LightGBM 参数说明：
    #   n_estimators: 树的数量
    #   max_depth: 最大深度
    #   learning_rate: 学习率
    #   verbose: 小于 0 关闭训练日志
    model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbose=-1)
    model.fit(Xtr, ytr)

    print("test acc:", accuracy_score(yte, model.predict(Xte)))
    # 特征重要性：基于分裂次数
    print("feature importance:", model.feature_importances_)


if __name__ == "__main__":
    main()
