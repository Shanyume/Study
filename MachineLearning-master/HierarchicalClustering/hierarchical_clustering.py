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

    # 对比三种常用的合并策略：
    #   ward：最小化合并后两簇的 SSE 增量（最小方差法），簇紧凑、大小相近
    #   complete：用两簇间最远点对的距离衡量簇间距离，倾向产生小而密的簇
    #   average：用两簇间所有点对的平均距离衡量，介于 single/complete 之间
    for linkage_method in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
        labels = model.fit_predict(X)
        # ARI：调整兰德指数，校正随机期望后的一致性指标，1 为完全一致
        print(f"linkage={linkage_method:>8}, ARI={adjusted_rand_score(y, labels):.3f}")

    # 绘制树状图（ward linkage）
    # linkage 自底向上合并：Z 每行记录一次合并（簇 a、簇 b、距离），
    # 样本索引用 0..n-1，合并出的新簇用 n..2n-2 编号
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
