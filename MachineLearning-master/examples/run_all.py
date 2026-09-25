#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一验证脚本：快速冒烟测试核心机器学习模块
=============================================

对项目核心模块做快速验证，确认在当前环境下能正常运行。
每个测试函数加载对应模块，用小数据集或合成数据运行，
通过 assert 检查输出是否符合预期（维度、准确率等）。

测试覆盖的模块：
    1. PCA — 降维和重构维度检查
    2. Decision Tree (ID3) — 训练集预测准确性
    3. KMeans — 标签数量和 SSE 合理性
    4. Naive Bayes — 多项式和高斯朴素贝叶斯预测
    5. SVM SMO — 训练集准确率 > 0.5
    6. SVM QP — 小样本线性可分预测
    7. Kernel Ridge — 预测数量检查
    8. Iris sklearn — KNN / LR / SVM 多模型对比

用法：python examples/run_all.py

依赖：numpy、pickle、csv、各模块自身的依赖
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使各模块可以被导入
# Path(__file__).resolve().parent.parent 即 examples/ 的上一级 = 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pickle
import csv


def check_pca():
    """
    PCA 冒烟测试

    测试内容：
        - 生成 100×10 的随机数据
        - 用 PCA 降维（保留 95% 方差）
        - 检查降维后维度 < 10（确实降维了）
        - 检查重构后维度 = 原始维度（重构正确）
    """
    from PCA.pca import pca
    # 生成 100 个样本、10 维特征的随机数据
    X = np.random.RandomState(42).randn(100, 10)
    # pca 返回 (降维后的数据, 重构的数据)
    # percent=0.95 表示保留 95% 的方差，自动选择主成分个数
    low, rec = pca(X, percent=0.95)
    assert low.shape[0] == 100 and low.shape[1] < 10, "PCA 输出维度异常"
    assert rec.shape == X.shape, "PCA 重构维度异常"
    print("PCA: OK")


def check_decision_tree():
    """
    决策树冒烟测试

    测试内容：
        - 用手工小数据集（5 个样本、5 个特征）训练 ID3 决策树
        - 检查训练集预测结果与真实标签完全一致
        - 验证 ID3 算法在简单数据上能完美拟合
    """
    from DecisionTree.id3_c45 import DecisionTree
    # 手工构造的小数据集：5 个样本，5 个离散特征
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']
    # 用 ID3 算法训练决策树（基于信息增益）
    clf = DecisionTree(mode='ID3')
    clf.fit(X, y)
    pred = clf.predict(X)
    # 决策树在训练集上应该能完美拟合（无剪枝时）
    assert list(pred) == y, "ID3 决策树训练集预测不匹配"
    print("Decision Tree: OK")


def check_kmeans():
    """
    KMeans 冒烟测试

    测试内容：
        - 加载 KMeans/data.pkl 中的聚类数据
        - 用 K=3 聚类，最多迭代 10 次
        - 检查标签数量 = 样本数量
        - 检查 SSE（误差平方和）> 0
    """
    from KMeans.kmeans import KMeans
    # 加载预制的聚类数据（特征矩阵 X 和真实标签 y）
    with open(ROOT / 'KMeans' / 'data.pkl', 'rb') as f:
        X, y = pickle.load(f, encoding='latin1')
    # 用 K=3 聚类，初始质心取 X[50:53]（3 个样本作为初始中心）
    clf = KMeans(n_clusters=3, initCent=X[50:53], max_iter=10)
    clf.fit(X)
    # 检查聚类结果
    assert clf.labels.shape[0] == X.shape[0], "KMeans 标签数量异常"
    assert clf.sse is not None and clf.sse > 0, "KMeans SSE 异常"
    print("KMeans: OK")


def check_naive_bayes():
    """
    朴素贝叶斯冒烟测试

    测试内容：
        - 用手工小数据集训练多项式朴素贝叶斯和高斯朴素贝叶斯
        - 检查多项式 NB 的预测类别在已知类别集合中
        - 检查高斯 NB 的预测数量 = 样本数量
    """
    from NaiveBayes.NaiveBayes import MultinomialNB, GaussianNB
    # 手工构造的二分类数据集：15 个样本，2 个离散特征
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ]).T
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    # 测试多项式朴素贝叶斯（适用于离散特征）
    nb = MultinomialNB(alpha=1.0, fit_prior=True)  # alpha=1.0 为拉普拉斯平滑
    nb.fit(X, y)
    pred = nb.predict(np.array([2, 4]))
    assert pred in nb.classes_, "多项式朴素贝叶斯预测类别异常"

    # 测试高斯朴素贝叶斯（假设特征服从高斯分布）
    gnb = GaussianNB()
    gpred = gnb.fit(X, y).predict(X)
    assert len(gpred) == len(X), "高斯朴素贝叶斯预测数量异常"
    print("Naive Bayes: OK")


def check_svm_smo():
    """
    SVM SMO 冒烟测试

    测试内容：
        - 加载 iris-slwc.txt（鸢尾花数据集的两个特征、两个类别）
        - 用 SMO 算法训练 SVM（最多 100 次迭代）
        - 检查训练集准确率 > 0.5（至少比随机好）
    """
    sys.path.insert(0, str(ROOT / 'SVM' / 'SVM_by_SMO'))
    from SVCSMO import SVCSMO
    # 读取 CSV 格式的鸢尾花数据
    data = []
    with open(ROOT / 'SVM' / 'SVM_by_SMO' / 'small_data' / 'iris-slwc.txt') as f:
        for row in csv.reader(f, delimiter=','):
            data.append([float(x) for x in row])
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1].astype(int)  # 最后一列为标签
    # 用 SMO（序列最小优化）算法训练 SVM
    model = SVCSMO(max_iter=100)
    model.fit(X, y)
    y_hat = model.predict(X)
    acc = np.mean(y_hat == y)  # 计算训练集准确率
    assert acc > 0.5, f"SVM SMO 准确率过低: {acc}"
    print(f"SVM SMO: OK (train accuracy={acc:.3f})")


def check_svm_qp():
    """
    SVM QP 冒烟测试

    测试内容：
        - 用 4 个样本的简单二分类数据（线性可分）
        - 用二次规划（QP）方法训练线性核 SVM
        - 检查预测结果与真实标签完全一致
    """
    sys.path.insert(0, str(ROOT / 'SVM' / 'SVM_by_QP'))
    from SVCQP import SVM, linear_kernel
    # 4 个样本的简单二分类数据：左下两个点为 -1，右上两个点为 +1
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([-1, -1, 1, 1])
    # 用线性核 + C=1.0 训练 SVM
    clf = SVM(kernel=linear_kernel, C=1.0)
    clf.fit(X, y)
    y_hat = clf.predict(X)
    # 线性可分数据应该能完美分类
    assert np.allclose(y_hat, y), "SVM QP 小样本预测异常"
    print("SVM QP: OK")


def check_kernel_ridge():
    """
    Kernel Ridge 冒烟测试

    测试内容：
        - 加载 iris-slwc.txt 数据
        - 用高斯核 + C=0.1 + gamma=5.0 训练 Kernel Ridge 回归
        - 检查预测数量 = 样本数量
    """
    sys.path.insert(0, str(ROOT / 'Ridge' / 'kernel_ridge'))
    from kernel_ridge import KernelRidge
    # 读取 CSV 格式的鸢尾花数据
    data = []
    with open(ROOT / 'Ridge' / 'kernel_ridge' / 'small_data' / 'iris-slwc.txt') as f:
        for row in csv.reader(f, delimiter=','):
            data.append([float(x) for x in row])
    data = np.array(data)
    X, y = data[:, :-1], data[:, -1].reshape(-1, 1)  # 标签 reshape 为列向量
    # 用高斯核（RBF）训练 Kernel Ridge 回归
    model = KernelRidge(kernel_type='gaussian', C=0.1, gamma=5.0)
    model.fit(X, y)
    y_hat = model.predict(X, X)
    assert y_hat.shape[0] == X.shape[0], "KRR 预测数量异常"
    print("Kernel Ridge: OK")


def check_iris_sklearn():
    """
    Iris sklearn 冒烟测试

    测试内容：
        - 用 sklearn 加载完整鸢尾花数据集
        - 划分 70% 训练集、30% 测试集
        - 训练 KNN、逻辑回归、SVM 三个模型
        - 检查每个模型的测试集准确率 > 0.8
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    # 加载鸢尾花数据集
    iris = load_iris()
    # 按 7:3 划分，stratify 保证各类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )
    # 定义三个分类模型（基于距离/梯度的模型用 Pipeline 封装标准化）
    models = {
        # KNN：先标准化，再 K=5 近邻
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        # 逻辑回归：先标准化，再 max_iter=200 保证收敛
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
        # SVM：先标准化，再 RBF 核
        'SVM': make_pipeline(StandardScaler(), SVC(kernel='rbf')),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        # Iris 数据集简单，准确率应该 > 0.8
        assert acc > 0.8, f"{name} 准确率过低: {acc}"
        print(f"Iris / {name}: OK (accuracy={acc:.3f})")


def main():
    """依次运行所有冒烟测试。"""
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
