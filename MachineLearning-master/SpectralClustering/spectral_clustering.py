#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral Clustering（谱聚类）
==============================

谱聚类把数据看作图：
    1. 构建相似度图（邻接矩阵）
    2. 计算图拉普拉斯矩阵的特征向量
    3. 在低维特征空间中做 KMeans

数据集：sklearn Iris
依赖：scikit-learn
运行：python spectral_clustering.py
"""
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


def main():
    """谱聚类演示。"""
    X, y = load_iris(return_X_y=True)
    # affinity="nearest_neighbors" 用 kNN 图构建相似度
    # 内部步骤：kNN 图 -> 拉普拉斯矩阵 L = D - W -> 取最小 n_clusters 个特征值
    # 对应的特征向量作为新特征，再在新特征上跑 KMeans
    # n_neighbors：kNN 图的邻居数，太大时图连通性过强、非凸簇结构被抹平
    model = SpectralClustering(n_clusters=3, affinity="nearest_neighbors", n_neighbors=10, random_state=42)
    labels = model.fit_predict(X)
    # ARI：调整兰德指数，衡量聚类结果与真实类别的一致性
    print("ARI:", adjusted_rand_score(y, labels))


if __name__ == "__main__":
    main()
