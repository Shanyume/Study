#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaBoost（自适应提升）
========================

AdaBoost 是经典的 Boosting 算法：
    1. 每轮迭代训练一个弱学习器（决策树桩）
    2. 根据该学习器的错误率更新样本权重：
       - 分错的样本权重增大，下轮更受关注
       - 分对的样本权重减小
    3. 根据错误率计算该学习器的权重 alpha
    4. 最终模型是所有弱学习器的加权投票

数据集：sklearn Iris（二分类）
依赖：numpy、scikit-learn
"""


import numpy as np


class DecisionStump:
    """
    决策树桩（单层决策树）
    AdaBoost 的弱学习器
    """

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # 1 或 -1，决定分类方向
        self.alpha = None  # 分类器权重

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        # 遍历所有特征和阈值，寻找最优决策树桩
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature_idx] <= threshold] = -1
                    else:
                        predictions[X[:, feature_idx] > threshold] = -1

                    # 计算加权错误率
                    error = np.sum(weights[predictions != y])

                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature_idx] <= self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] > self.threshold] = -1

        return predictions


class AdaBoost:
    """
    AdaBoost 分类器
    通过迭代调整样本权重，聚焦于难以分类的样本
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        # 初始化样本权重，均匀分布
        weights = np.ones(n_samples) / n_samples

        self.stumps = []

        for _ in range(self.n_estimators):
            # 训练决策树桩
            stump = DecisionStump()
            stump.fit(X, y, weights)

            # 计算预测值
            predictions = stump.predict(X)

            # 计算加权错误率
            error = np.sum(weights[predictions != y])

            # 计算分类器权重（alpha）
            # alpha = 0.5 * ln((1-error)/error)，错误率越低权重越大
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            stump.alpha = alpha

            # 更新样本权重
            # 正确分类的样本权重降低，错误分类的样本权重升高
            for i in range(n_samples):
                if predictions[i] == y[i]:
                    weights[i] *= np.exp(-alpha)
                else:
                    weights[i] *= np.exp(alpha)

            # 归一化权重
            weights /= np.sum(weights)

            self.stumps.append(stump)

    def predict(self, X):
        # 加权投票
        stump_predictions = np.array([stump.predict(X) * stump.alpha
                                      for stump in self.stumps])
        # 求和并取符号
        return np.sign(np.sum(stump_predictions, axis=0))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    演示 AdaBoost 分类
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("AdaBoost 分类演示")
    print("=" * 50)

    # 生成模拟数据（二分类，标签为 -1 和 1）
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                               n_informative=10, random_state=42)
    # 将标签转换为 -1 和 1
    y = np.where(y == 0, -1, 1)  # # 标签转为 {-1, 1}

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练 AdaBoost
    ada = AdaBoost(n_estimators=50)
    ada.fit(X_train, y_train)

    # 评估
    train_acc = ada.score(X_train, y_train)
    test_acc = ada.score(X_test, y_test)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"弱分类器数量: {len(ada.stumps)}")

    # 展示前 5 个分类器的权重
    print("\n前 5 个弱分类器权重 (alpha):")
    for i, stump in enumerate(ada.stumps[:5]):
        print(f"  分类器 {i+1}: alpha = {stump.alpha:.4f}")


if __name__ == '__main__':
    demo()
