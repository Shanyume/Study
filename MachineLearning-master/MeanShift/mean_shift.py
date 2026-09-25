#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean Shift 聚类
================

Mean Shift 是基于密度峰值的聚类算法：
    - 每个点向密度梯度方向移动
    - 最终收敛到密度峰值，即为簇中心
    - 不需要指定簇数，自动确定

数据集：sklearn Iris
依赖：scikit-learn
运行：python mean_shift.py
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score


def main():
    """Mean Shift 聚类演示。"""
    X, y = load_iris(return_X_y=True)

    # 自动估计带宽参数（决定邻域大小）
    # bandwidth 是核半径：太大时不同簇的密度盆合并、簇数偏少；太小则噪声敏感、簇数偏多
    # quantile 越小，选出的带宽越小
    bandwidth = estimate_bandwidth(X, quantile=0.3, random_state=42)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # bin_seeding=True：只把网格（bin）中心作为移动起点，大幅减少迭代次数
    labels = model.fit_predict(X)
    # 簇数由数据本身的密度结构决定：收敛到不同密度峰值的点构成一个簇

    print("n_clusters:", len(model.cluster_centers_))  # 自动发现的簇数
    # ARI：调整兰德指数，聚类标签与真实标签的一致性
    print("ARI:", adjusted_rand_score(y, labels))


if __name__ == "__main__":
    main()
