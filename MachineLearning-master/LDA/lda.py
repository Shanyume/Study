#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class LDA:
    """
    线性判别分析（Linear Discriminant Analysis）
    用于降维和分类，最大化类间方差与类内方差的比值
    """

    def __init__(self, n_components=None):
        """
        :param n_components: 降维后的维度，默认为类别数 - 1
        """
        self.n_components = n_components
        self.components = None  # 投影矩阵
        self.means = None       # 各类别均值
        self.classes = None     # 类别标签

    def fit(self, X, y):
        """
        训练 LDA 模型
        :param X: 特征矩阵，形状 (n_samples, n_features)
        :param y: 类别标签，形状 (n_samples,)
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes = np.unique(y)
        n_features = X.shape[1]

        # 默认降维到类别数 - 1 维
        if self.n_components is None:
            self.n_components = len(self.classes) - 1

        # 计算类内散度矩阵 (Within-class scatter matrix)
        # S_W = Σ Σ (x - μ_k)(x - μ_k)^T，其中 k 遍历所有类别
        S_W = np.zeros((n_features, n_features))
        self.means = []

        for c in self.classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.means.append(mean_c)

            # 累加每个类别的散度
            diff = X_c - mean_c
            S_W += diff.T @ diff

        # 计算类间散度矩阵 (Between-class scatter matrix)
        # S_B = Σ n_k (μ_k - μ)(μ_k - μ)^T，其中 μ 是总体均值
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((n_features, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_diff = (self.means[i] - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 求解广义特征值问题：S_W^(-1) S_B w = λ w
        # 取前 n_components 个最大特征值对应的特征向量
        try:
            S_W_inv = np.linalg.inv(S_W)
            matrix = S_W_inv @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            # 按特征值降序排列
            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            self.components = eigenvectors[:, sorted_indices[:self.n_components]].real
        except np.linalg.LinAlgError:
            # 如果 S_W 不可逆，使用伪逆
            S_W_pinv = np.linalg.pinv(S_W)
            matrix = S_W_pinv @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            sorted_indices = np.argsort(eigenvalues.real)[::-1]
            self.components = eigenvectors[:, sorted_indices[:self.n_components]].real

    def transform(self, X):
        """
        将数据投影到低维空间
        :param X: 特征矩阵
        :return: 降维后的数据
        """
        return X @ self.components

    def fit_transform(self, X, y):
        """
        拟合并转换数据
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        使用 LDA 进行分类（基于投影后的最近质心）
        """
        X_transformed = self.transform(X)

        # 计算投影后各类别的质心
        class_means_transformed = np.array([self.transform(np.array([mean]))[0]
                                            for mean in self.means])

        # 对每个样本，找到最近的质心
        predictions = []
        for x in X_transformed:
            distances = np.sqrt(np.sum((class_means_transformed - x) ** 2, axis=1))
            predictions.append(self.classes[np.argmin(distances)])

        return np.array(predictions)

    def score(self, X, y):
        """
        计算分类准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    演示 LDA 降维和分类
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("线性判别分析 (LDA) 演示")
    print("=" * 50)

    # 加载鸢尾花数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"数据集: {len(X)} 样本, {X.shape[1]} 特征, {len(np.unique(y))} 类别")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练 LDA（降维到 2 维）
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    # 分类准确率
    train_acc = lda.score(X_train, y_train)
    test_acc = lda.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 可视化降维结果
    plt.figure(figsize=(10, 5))

    # 左图：原始数据（取前两个特征）
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green']
    for i, c in enumerate(np.unique(y)):
        mask = y == c
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=iris.target_names[c], alpha=0.7)
    plt.title('Original Data (first 2 features)')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：LDA 降维后
    plt.subplot(1, 2, 2)
    for i, c in enumerate(np.unique(y_train)):
        mask = y_train == c
        plt.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1],
                    c=colors[i], label=iris.target_names[c], alpha=0.7)
    plt.title('LDA Projection (2D)')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lda_projection.png', dpi=150)
    print("\nPlot saved: lda_projection.png")
    plt.close()


if __name__ == '__main__':
    demo()
