#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次聚类（Agglomerative Clustering）
======================================

层次聚类从每个样本各自为一簇开始，逐步合并距离最近的簇，
最终形成树状结构。不需要预先指定簇数（可指定）。

数据集：sklearn Iris
依赖：scikit-learn、scipy、matplotlib
运行：python hierarchical_clustering.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path

# 图片保存目录
BASE = Path(__file__).resolve().parent


def main():
    """层次聚类：对比不同 linkage 方法并绘制树状图。"""
    X, y = load_iris(return_X_y=True)

    # 对比三种常用的合并策略
    for linkage_method in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
        labels = model.fit_predict(X)
        print(f"linkage={linkage_method:>8}, ARI={adjusted_rand_score(y, labels):.3f}")

    # 绘制树状图（ward linkage）
    Z = linkage(X, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)  # 截断到前 5 层
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(BASE / "dendrogram.png", dpi=120)
    print("dendrogram saved:", BASE / "dendrogram.png")


if __name__ == "__main__":
    main()
