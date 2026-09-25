#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest（随机森林）
==========================

随机森林是决策树的 Bagging 集成：
    - 自助采样训练多棵 CART 决策树
    - 每棵树分裂时随机选择特征子集
    - 预测时所有树投票

本脚本包含 CART 决策树和随机森林实现。

依赖：numpy
"""
import numpy as np
from collections import Counter


class DecisionTree:
    """
    决策树分类器（CART 算法）
    使用基尼系数作为分裂标准
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件：达到最大深度、样本数过少、纯度足够
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}

        # 寻找最佳分裂特征和阈值
        best_feature, best_threshold, best_gini = None, None, float('inf')

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gini = self._calculate_gini(X, y, feature_idx, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        # 无法找到有效分裂
        if best_feature is None:
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}

        # 分裂数据集
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _calculate_gini(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return float('inf')

        gini_left = 1 - np.sum([(np.sum(y[left_mask] == c) / np.sum(left_mask)) ** 2
                                for c in np.unique(y)])
        gini_right = 1 - np.sum([(np.sum(y[right_mask] == c) / np.sum(right_mask)) ** 2
                                 for c in np.unique(y)])

        n = len(y)
        gini = (np.sum(left_mask) / n) * gini_left + (np.sum(right_mask) / n) * gini_right
        return gini

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['label']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])


class RandomForest:
    """
    随机森林分类器
    通过构建多棵决策树并投票来提高泛化能力
    """

    def __init__(self, n_trees=100, max_depth=5, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 每棵树使用的特征数，默认 sqrt(n_features)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        # 默认使用 sqrt(n_features) 个特征
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_trees):
            # Bootstrap 采样：有放回地随机抽取样本
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 随机选择特征子集
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_bootstrap[:, feature_indices]

            # 训练决策树
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_bootstrap)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # 每棵树投票
        predictions = np.array([tree.predict(X[:, features])
                                for tree, features in self.trees])
        # 多数投票
        return np.array([Counter(col).most_common(1)[0][0]
                         for col in predictions.T])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    演示随机森林分类
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("随机森林分类演示")
    print("=" * 50)

    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练随机森林
    rf = RandomForest(n_trees=50, max_depth=10)
    rf.fit(X_train, y_train)

    # 评估
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"决策树数量: {len(rf.trees)}")
    print(f"每棵树特征数: {rf.max_features}")


if __name__ == '__main__':
    demo()
