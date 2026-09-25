#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 分类示例
==================

XGBoost 是工业界最常用的梯度提升树实现，支持正则化、列采样、并行等特性。

数据集：sklearn Iris（3 分类）
依赖：xgboost、scikit-learn
运行：python xgboost_example.py
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


def main():
    """训练 XGBoost 并输出准确率、分类报告和特征重要性。"""
    # 加载 Iris 数据集
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # XGBoost 参数说明：
    #   n_estimators: 树的数量
    #   max_depth: 每棵树最大深度，控制复杂度
    #   learning_rate: 学习率，越小越保守
    #   subsample: 每棵树采样比例（行采样）
    #   colsample_bytree: 每棵树特征采样比例（列采样）
    # 默认目标函数为 multi:softmax（多分类 softmax），
    # eval_metric="mlogloss" 即多分类 log 损失，用于训练过程中的评估
    model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss"
    )
    model.fit(Xtr, ytr)

    # 预测和评估
    # classification_report 给出每个类别的 precision / recall / F1 分数
    pred = model.predict(Xte)
    print("test acc:", accuracy_score(yte, pred))
    print(classification_report(yte, pred))

    # 特征重要性：每个特征对模型贡献度
    # 默认 importance_type="split"，按各特征被选为分裂点的次数统计
    print("feature importance:", model.feature_importances_)


if __name__ == "__main__":
    main()
