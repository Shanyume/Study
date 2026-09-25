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
    model = SpectralClustering(n_clusters=3, affinity="nearest_neighbors", n_neighbors=10, random_state=42)
    labels = model.fit_predict(X)
    print("ARI:", adjusted_rand_score(y, labels))


if __name__ == "__main__":
    main()
