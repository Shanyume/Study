#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM 高斯混合模型（EM 算法）
============================

GMM 假设数据由 k 个高斯分布混合而成，用 EM 算法迭代估计参数：
    - E-step：计算每个样本属于每个高斯分量的响应度（软分配）
    - M-step：用响应度加权更新均值、协方差和混合系数

与 KMeans 相比，GMM 是软聚类（输出概率），协方差可以是椭圆形状。

数据集：sklearn Iris
依赖：numpy、scipy、scikit-learn（仅用于数据加载和评估）
运行：python gmm.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal


class GMM:
    """高斯混合模型，EM 算法实现。"""

    def __init__(self, n_components=3, max_iter=100, tol=1e-4, seed=42):
        """
        :param n_components: 高斯分量数（簇数）
        :param max_iter: 最大迭代次数
        :param tol: 对数似然变化阈值，小于则提前停止
        :param seed: 随机种子
        """
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        """EM 算法训练 GMM。"""
        X = np.asarray(X)
        rng = np.random.RandomState(self.seed)
        n, d = X.shape

        # 初始化：随机选 k 个点作为均值，协方差为单位阵
        self.mu = X[rng.choice(n, self.k, replace=False)]
        self.cov = np.array([np.eye(d) for _ in range(self.k)])
        self.pi = np.ones(self.k) / self.k  # 混合系数均匀初始化

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # ---- E-step：计算响应度 ----
            resp = np.zeros((n, self.k))
            for i in range(self.k):
                # 第 i 个高斯分量下每个样本的概率 × 混合系数
                resp[:, i] = self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i])
            # 归一化为后验概率（每行和为 1）
            resp /= resp.sum(axis=1, keepdims=True) + 1e-12

            # ---- M-step：更新参数 ----
            Nk = resp.sum(axis=0)  # 每个分量的有效样本数
            self.pi = Nk / n  # 更新混合系数
            self.mu = (resp.T @ X) / Nk[:, None]  # 更新均值
            for i in range(self.k):
                diff = X - self.mu[i]
                # 加权协方差 + 正则化项防止奇异
                self.cov[i] = (resp[:, i][:, None] * diff).T @ diff / Nk[i] + 1e-6*np.eye(d)

            # 计算对数似然，判断收敛
            densities = np.array([
                self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i])
                for i in range(self.k)])
            ll = np.sum(np.log(densities.sum(axis=0) + 1e-12))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.resp = resp
        return self

    def predict(self, X):
        """返回每个样本最可能的高斯分量索引。"""
        X = np.asarray(X)
        probs = np.array([
            self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i])
            for i in range(self.k)])
        return probs.argmax(axis=0)


def best_label_mapping(y_true, y_pred, k):
    """
    聚类标签和真实标签没有固定对应关系，需要做最优排列映射后计算准确率。
    :param y_true: 真实标签
    :param y_pred: 聚类标签
    :param k: 簇数
    :return: 最优映射后的准确率
    """
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in y_pred])
        acc = accuracy_score(y_true, mapped)
        best_acc = max(best_acc, acc)
    return best_acc


if __name__ == "__main__":
    from KMeans.kmeans import KMeans
    X, y = load_iris(return_X_y=True)
    # GMM 软聚类
    gmm = GMM(n_components=3).fit(X)
    print("GMM acc:", best_label_mapping(y, gmm.predict(X), 3))
    # KMeans 硬聚类对比
    km = KMeans(n_clusters=3, max_iter=50)
    km.fit(X)
    print("KMeans acc:", best_label_mapping(y, km.labels.astype(int), 3))
