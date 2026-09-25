#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性判别分析（Linear Discriminant Analysis, LDA）
===================================================

LDA 是一种经典的有监督降维和分类方法，核心思想是：
    找到一个投影方向，使得投影后：
    - 类间散度（Between-class scatter）尽量大 → 不同类别的中心尽量远离
    - 类内散度（Within-class scatter）尽量小 → 同一类别的样本尽量聚集

数学原理：
    1. 类内散度矩阵 S_W = Σ_k Σ_{x∈C_k} (x - μ_k)(x - μ_k)^T
       - 衡量每个类别内部的分散程度
       - μ_k 是第 k 类的均值向量

    2. 类间散度矩阵 S_B = Σ_k n_k (μ_k - μ)(μ_k - μ)^T
       - 衡量不同类别中心之间的分散程度
       - μ 是全体样本的均值，n_k 是第 k 类的样本数

    3. 优化目标：最大化 tr(W^T S_B W) / tr(W^T S_W W)
       - 等价于求解广义特征值问题：S_W^{-1} S_B w = λ w
       - 取前 n_components 个最大特征值对应的特征向量作为投影方向

    4. 投影：X_lda = X @ W，其中 W 是前 n_components 个特征向量组成的矩阵

    5. 分类：在投影空间中，用最近质心法分类
       - 计算每个类别在投影空间的质心
       - 对每个样本，找到距离最近的质心，其类别即为预测类别

与 PCA 的区别：
    - PCA 是无监督的，只保留全局方差最大的方向
    - LDA 是有监督的，保留类别可分性最好的方向
    - LDA 最多降维到 C-1 维（C 为类别数），PCA 可降维到任意维度

依赖：numpy、matplotlib、sklearn（仅 demo 用）
运行：python lda.py
"""

import numpy as np
import matplotlib.pyplot as plt


class LDA:
    """
    线性判别分析（Linear Discriminant Analysis）

    用于降维和分类，最大化类间方差与类内方差的比值

    属性：
        n_components: 降维后的目标维度
        components: 投影矩阵 (n_features, n_components)，每列是一个投影方向
        means: 各类别在原始空间的均值向量列表
        classes: 类别标签数组
    """

    def __init__(self, n_components=None):
        """
        初始化 LDA

        :param n_components: 降维后的维度，默认为类别数 - 1
                             （因为 C 个类别最多只有 C-1 个判别方向）
        """
        self.n_components = n_components
        self.components = None  # 投影矩阵，fit 后填充
        self.means = None       # 各类别均值向量列表，fit 后填充
        self.classes = None     # 类别标签数组，fit 后填充

    def fit(self, X, y):
        """
        训练 LDA 模型：计算类内散度矩阵 S_W 和类间散度矩阵 S_B，
        求解广义特征值问题，取前 n_components 个最大特征值对应的特征向量

        :param X: 特征矩阵，形状 (n_samples, n_features)
        :param y: 类别标签，形状 (n_samples,)
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        # 提取所有不重复的类别标签
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # 如果未指定降维维度，默认降维到 C-1 维（C 为类别数）
        # 因为 C 个类别的均值向量最多张成 C-1 维子空间
        if self.n_components is None:
            self.n_components = len(self.classes) - 1

        # ---- 第一步：计算类内散度矩阵 S_W ----
        # S_W = Σ_k Σ_{x∈C_k} (x - μ_k)(x - μ_k)^T
        # 物理含义：所有类别内部的协方差之和，衡量类内分散程度
        S_W = np.zeros((n_features, n_features))
        self.means = []

        for c in self.classes:
            # 取出当前类别的所有样本
            X_c = X[y == c]
            # 计算当前类别的均值向量 μ_k
            mean_c = np.mean(X_c, axis=0)
            self.means.append(mean_c)

            # 计算每个样本与类均值的差 (x - μ_k)
            diff = X_c - mean_c
            # 累加 (x - μ_k)(x - μ_k)^T，即当前类别的散度矩阵
            # diff.T @ diff 等价于 Σ_i (x_i - μ_k)(x_i - μ_k)^T
            S_W += diff.T @ diff

        # ---- 第二步：计算类间散度矩阵 S_B ----
        # S_B = Σ_k n_k (μ_k - μ)(μ_k - μ)^T
        # 物理含义：各类别均值与总体均值的偏差加权平方和，衡量类间分散程度
        overall_mean = np.mean(X, axis=0)  # 全体样本的均值 μ
        S_B = np.zeros((n_features, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]  # 当前类别的样本数 n_k
            # (μ_k - μ) 转为列向量
            mean_diff = (self.means[i] - overall_mean).reshape(-1, 1)
            # 累加 n_k × (μ_k - μ)(μ_k - μ)^T
            S_B += n_c * (mean_diff @ mean_diff.T)

        # ---- 第三步：求解广义特征值问题 S_W^{-1} S_B w = λ w ----
        # 目标：找到使 S_B / S_W 比值最大的投影方向
        # 等价于求 S_W^{-1} S_B 的特征值和特征向量，取前 k 个最大特征值对应的特征向量
        try:
            # 尝试直接求 S_W 的逆矩阵
            S_W_inv = np.linalg.inv(S_W)
            matrix = S_W_inv @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            # 按特征值降序排列（特征值越大，对应的投影方向判别能力越强）
            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            # 取前 n_components 个特征向量作为投影矩阵
            self.components = eigenvectors[:, sorted_indices[:self.n_components]].real
        except np.linalg.LinAlgError:
            # 如果 S_W 不可逆（奇异矩阵），使用伪逆（Moore-Penrose 伪逆）
            # 这种情况通常发生在特征维度 > 样本数，或存在线性相关的特征
            S_W_pinv = np.linalg.pinv(S_W)
            matrix = S_W_pinv @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            self.components = eigenvectors[:, sorted_indices[:self.n_components]].real

    def transform(self, X):
        """
        将数据投影到低维空间

        投影公式：X_lda = X @ W
        其中 W 是 fit 阶段求得的投影矩阵 (n_features, n_components)

        :param X: 特征矩阵 (n_samples, n_features)
        :return: 降维后的数据 (n_samples, n_components)
        """
        return X @ self.components

    def fit_transform(self, X, y):
        """
        先训练（fit）再投影（transform），一步完成降维

        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 类别标签 (n_samples,)
        :return: 降维后的数据 (n_samples, n_components)
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        使用 LDA 进行分类（基于投影后的最近质心法）

        分类策略：
            1. 将训练集的各类别均值向量投影到低维空间
            2. 对每个测试样本，也投影到低维空间
            3. 计算测试样本到各类别质心的欧氏距离
            4. 选择距离最近的质心对应的类别作为预测结果

        :param X: 待预测的特征矩阵 (n_samples, n_features)
        :return: 预测的类别标签数组 (n_samples,)
        """
        # 将待预测样本投影到低维空间
        X_transformed = self.transform(X)

        # 将训练阶段保存的各类别均值向量也投影到低维空间
        # 这些投影后的均值就是各类别在低维空间的"质心"
        class_means_transformed = np.array([self.transform(np.array([mean]))[0]
                                            for mean in self.means])

        # 对每个样本，找到距离最近的质心
        predictions = []
        for x in X_transformed:
            # 计算当前样本到各类别质心的欧氏距离
            distances = np.sqrt(np.sum((class_means_transformed - x) ** 2, axis=1))
            # 选择距离最小的质心对应的类别
            predictions.append(self.classes[np.argmin(distances)])

        return np.array(predictions)

    def score(self, X, y):
        """
        计算分类准确率

        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 真实类别标签 (n_samples,)
        :return: 准确率（0~1 之间）
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    演示 LDA 在鸢尾花数据集上的降维和分类效果

    流程：
        1. 加载 Iris 数据集（150 样本，4 特征，3 类别）
        2. 划分训练集（70%）和测试集（30%）
        3. 用 LDA 降维到 2 维
        4. 计算训练集和测试集的分类准确率
        5. 可视化：左图为原始数据前两个特征，右图为 LDA 降维后的 2D 散点图
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("线性判别分析 (LDA) 演示")
    print("=" * 50)

    # 加载鸢尾花数据集：150 个样本，4 个特征（萼片长/宽、花瓣长/宽），3 个类别
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"数据集: {len(X)} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")

    # 按 7:3 划分训练集和测试集，stratify 保证各类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练 LDA 模型，降维到 2 维（3 类问题最多可降维到 2 维）
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)  # 在训练集上拟合并投影
    X_test_lda = lda.transform(X_test)                  # 用训练好的模型投影测试集

    # 计算分类准确率（基于投影后的最近质心法）
    train_acc = lda.score(X_train, y_train)
    test_acc = lda.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # ---- 可视化 ----
    plt.figure(figsize=(10, 5))

    # 左图：原始数据（取前两个特征：萼片长度 vs 萼片宽度）
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green']
    for i, c in enumerate(np.unique(y)):
        mask = y == c  # 当前类别的布尔掩码
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=iris.target_names[c], alpha=0.7)
    plt.title('Original Data (first 2 features)')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：LDA 降维后的 2D 散点图
    # 可以看到 LDA 投影后三个类别的分离度比原始特征更好
    plt.subplot(1, 2, 2)
    for i, c in enumerate(np.unique(y_train)):
        mask = y_train == c
        plt.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1],
                    c=colors[i], label=iris.target_names[c], alpha=0.7)
    plt.title('LDA Projection (2D)')
    plt.xlabel('LD1')  # 第一判别方向
    plt.ylabel('LD2')  # 第二判别方向
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lda_projection.png', dpi=150)
    print("\nPlot saved: lda_projection.png")
    plt.close()


if __name__ == '__main__':
    demo()
