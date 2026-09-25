#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一验证脚本：快速冒烟测试核心机器学习模块
=============================================

对项目核心模块做快速验证，确认在当前环境下能正常运行：
- PCA、Decision Tree、KMeans、Naive Bayes
- SVM SMO、SVM QP、Kernel Ridge
- Iris sklearn 多模型对比

用法：python examples/run_all.py
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，方便导入各模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pickle
import csv


def check_pca():
    """PCA 冒烟测试"""
    from PCA.pca import pca
    X = np.random.RandomState(42).randn(100, 10)
    low, rec = pca(X, percent=0.95)
    assert low.shape[0] == 100 and low.shape[1] < 10, "PCA 输出维度异常"
    assert rec.shape == X.shape, "PCA 重构维度异常"
    print("PCA: OK")


def check_decision_tree():
    """决策树冒烟测试"""
    from DecisionTree.id3_c45 import DecisionTree
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']
    clf = DecisionTree(mode='ID3')
    clf.fit(X, y)
    pred = clf.predict(X)
    assert list(pred) == y, "ID3 决策树训练集预测不匹配"
    print("Decision Tree: OK")


def check_kmeans():
    """KMeans 冒烟测试"""
    from KMeans.kmeans import KMeans
    with open(ROOT / 'KMeans' / 'data.pkl', 'rb') as f:
        X, y = pickle.load(f, encoding='latin1')
    clf = KMeans(n_clusters=3, initCent=X[50:53], max_iter=10)
    clf.fit(X)
    assert clf.labels.shape[0] == X.shape[0], "KMeans 标签数量异常"
    assert clf.sse is not None and clf.sse > 0, "KMeans SSE 异常"
    print("KMeans: OK")


def check_naive_bayes():
    """朴素贝叶斯冒烟测试"""
    from NaiveBayes.NaiveBayes import MultinomialNB, GaussianNB
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ]).T
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    nb = MultinomialNB(alpha=1.0, fit_prior=True)
    nb.fit(X, y)
    pred = nb.predict(np.array([2, 4]))
    assert pred in nb.classes_, "多项式朴素贝叶斯预测类别异常"
    gnb = GaussianNB()
    gpred = gnb.fit(X, y).predict(X)
    assert len(gpred) == len(X), "高斯朴素贝叶斯预测数量异常"
    print("Naive Bayes: OK")


def check_svm_smo():
    """SVM SMO 冒烟测试"""
    sys.path.insert(0, str(ROOT / 'SVM' / 'SVM_by_SMO'))
    from SVCSMO import SVCSMO
    data = []
    with open(ROOT / 'SVM' / 'SVM_by_SMO' / 'small_data' / 'iris-slwc.txt') as f:
        for row in csv.reader(f, delimiter=','):
            data.append([float(x) for x in row])
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1].astype(int)
    model = SVCSMO(max_iter=100)
    model.fit(X, y)
    y_hat = model.predict(X)
    acc = np.mean(y_hat == y)
    assert acc > 0.5, f"SVM SMO 准确率过低: {acc}"
    print(f"SVM SMO: OK (train accuracy={acc:.3f})")


def check_svm_qp():
    """SVM QP 冒烟测试"""
    sys.path.insert(0, str(ROOT / 'SVM' / 'SVM_by_QP'))
    from SVCQP import SVM, linear_kernel
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([-1, -1, 1, 1])
    clf = SVM(kernel=linear_kernel, C=1.0)
    clf.fit(X, y)
    y_hat = clf.predict(X)
    assert np.allclose(y_hat, y), "SVM QP 小样本预测异常"
    print("SVM QP: OK")


def check_kernel_ridge():
    """Kernel Ridge 冒烟测试"""
    sys.path.insert(0, str(ROOT / 'Ridge' / 'kernel_ridge'))
    from kernel_ridge import KernelRidge
    data = []
    with open(ROOT / 'Ridge' / 'kernel_ridge' / 'small_data' / 'iris-slwc.txt') as f:
        for row in csv.reader(f, delimiter=','):
            data.append([float(x) for x in row])
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1].reshape(-1, 1)
    model = KernelRidge(kernel_type='gaussian', C=0.1, gamma=5.0)
    model.fit(X, y)
    y_hat = model.predict(X, X)
    assert y_hat.shape[0] == X.shape[0], "KRR 预测数量异常"
    print("Kernel Ridge: OK")


def check_iris_sklearn():
    """Iris sklearn 冒烟测试：只验证几个核心 sklearn 分类器"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    models = {
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf')),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.8, f"{name} 准确率过低: {acc}"
        print(f"Iris / {name}: OK (accuracy={acc:.3f})")


def main():
    print("=" * 50)
    print("MachineLearning 统一冒烟测试")
    print("=" * 50)
    check_pca()
    check_decision_tree()
    check_kmeans()
    check_naive_bayes()
    check_svm_smo()
    check_svm_qp()
    check_kernel_ridge()
    check_iris_sklearn()
    print("\n全部核心模块验证通过。")


if __name__ == '__main__':
    main()
