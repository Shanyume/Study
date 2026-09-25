#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class DBSCAN:
    """
    DBSCAN 聚类算法
    基于密度的聚类，能发现任意形状的簇并识别噪声点
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        :param eps: 邻域半径，定义"附近"的距离阈值
        :param min_samples: 核心点所需的最小邻居数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # 聚类标签，-1 表示噪声点

    def fit(self, X):
        """
        执行 DBSCAN 聚类
        :param X: 数据矩阵，形状 (n_samples, n_features)
        :return: 聚类标签数组
        """
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)  # 初始化为噪声

        # 计算所有点对之间的距离矩阵
        distances = self._compute_distance_matrix(X)

        # 找出每个点的 eps-邻域
        neighborhoods = []
        for i in range(n_samples):
            neighbors = np.where(distances[i] <= self.eps)[0]  # # 找出 eps 邻域内所有点的索引
            neighborhoods.append(neighbors)

        cluster_id = 0

        for i in range(n_samples):
            # 跳过已处理的点
            if self.labels[i] != -1:
                continue

            # 检查是否为核心点
            if len(neighborhoods[i]) < self.min_samples:
                continue  # 噪声点，暂时标记为 -1

            # 开始新簇
            self._expand_cluster(i, neighborhoods, cluster_id)
            cluster_id += 1

        return self.labels

    def _expand_cluster(self, point_idx, neighborhoods, cluster_id):
        """
        从核心点扩展簇
        """
        # 使用队列进行广度优先搜索
        queue = [point_idx]
        self.labels[point_idx] = cluster_id

        while queue:
            current = queue.pop(0)
            neighbors = neighborhoods[current]

            # 如果当前点是核心点，将其邻居加入队列
            if len(neighbors) >= self.min_samples:
                for neighbor in neighbors:
                    if self.labels[neighbor] == -1:
                        self.labels[neighbor] = cluster_id
                        queue.append(neighbor)

    def _compute_distance_matrix(self, X):
        """
        计算欧氏距离矩阵
        :param X: 数据矩阵，形状 (n_samples, n_features)
        :return: 距离矩阵，形状 (n_samples, n_samples)
        """
        # 使用 broadcasting 计算距离：||x - y||² = ||x||² + ||y||² - 2*x·y
        sq_norms = np.sum(X ** 2, axis=1)
        distances = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
        # 处理数值误差
        distances = np.maximum(distances, 0)
        return np.sqrt(distances)

    def get_cluster_count(self):
        """获取簇的数量（不包括噪声）"""
        return len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def get_noise_count(self):
        """获取噪声点数量"""
        return np.sum(self.labels == -1)


def generate_cluster_data():
    """
    生成用于演示的聚类数据（月牙形 + 噪声）
    """
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # 添加一些噪声点
    noise = np.random.uniform(low=-1, high=2, size=(30, 2))
    X = np.vstack([X, noise])

    return X


def demo():
    """
    演示 DBSCAN 聚类
    """
    print("=" * 50)
    print("DBSCAN 聚类演示")
    print("=" * 50)

    X = generate_cluster_data()

    # 训练 DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit(X)

    n_clusters = dbscan.get_cluster_count()
    n_noise = dbscan.get_noise_count()

    print(f"发现簇数: {n_clusters}")
    print(f"噪声点数: {n_noise}")
    print(f"总样本数: {len(X)}")

    # 可视化
    plt.figure(figsize=(10, 5))

    # 左图：原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.7)
    plt.title('Original Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)

    # 右图：聚类结果
    plt.subplot(1, 2, 2)

    # 绘制簇
    unique_labels = set(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):  # # 遍历每个簇并分配颜色
        mask = labels == label
        if label == -1:
            # 噪声点用黑色 x 标记
            plt.scatter(X[mask, 0], X[mask, 1], s=20, c='black', marker='x', label='Noise')
        else:
            plt.scatter(X[mask, 0], X[mask, 1], s=20, c=[color], label=f'Cluster {label}')

    plt.title(f'DBSCAN Clustering (eps=0.3, min_samples=5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dbscan_clustering.png', dpi=150)
    print("\nPlot saved: dbscan_clustering.png")
    plt.close()


if __name__ == '__main__':
    demo()
