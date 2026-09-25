#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模型评估脚本
==================

对比多个分类器的 ROC 曲线和 PR 曲线，用于评估二分类模型性能。

- ROC 曲线：不同阈值下 TPR vs FPR，AUC 越接近 1 越好
- PR 曲线：Precision vs Recall，类别不平衡时更有参考价值

数据集：sklearn breast_cancer（二分类）
依赖：scikit-learn、numpy、matplotlib
运行：python model_evaluation.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path

# 图片保存目录：脚本所在目录
BASE = Path(__file__).resolve().parent


def main():
    """训练 3 个模型并绘制 ROC / PR 曲线。"""
    # 加载乳腺癌二分类数据集
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # 三个待比较的模型：LR / RF / SVM
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }

    # 左图：ROC 曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        model.fit(Xtr, ytr)
        # 取正类概率作为打分
        scores = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, scores)
        print(f"{name}: ROC-AUC={auc(fpr, tpr):.4f}")
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    # 随机猜测基准线
    plt.plot([0,1],[0,1],"k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend()

    # 右图：PR 曲线
    plt.subplot(1, 2, 2)
    for name, model in models.items():
        scores = model.predict_proba(Xte)[:, 1]
        prec, rec, _ = precision_recall_curve(yte, scores)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR Curve"); plt.legend()

    plt.tight_layout()
    plt.savefig(BASE / "roc_pr_curves.png", dpi=120)
    print("figure saved:", BASE / "roc_pr_curves.png")


if __name__ == "__main__":
    main()
