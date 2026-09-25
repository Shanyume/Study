#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest（随机森林）
==========================

随机森林是决策树的 Bagging 集成，核心思想：

    1. 自助采样（Bootstrap）：从原始训练集有放回抽取 n 个样本
    2. 每棵 CART 决策树只用该采样训练
    3. 每次分裂时随机选择部分特征（而非全部特征），增加树之间差异
    4. 分类时所有树投票，取多数作为最终预测
    5. 通过 Bagging 降低方差、提高泛化能力

CART 决策树：
    - 二叉分裂（每个节点只分左右两支）
    - 用基尼系数（Gini impurity）作为分裂标准
    - 支持连续特征阈值分裂

本脚本包含：
    - DecisionTree：CART 决策树（基尼系数）
    - RandomForest：随机森林集成

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
        # max_depth：限制树的深度，防止过拟合
        # min_samples_split：节点样本数低于该值时不再分裂
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # 递归建树：对当前节点样本集寻找最优二元分裂，再分别递归左右子集
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件：达到最大深度、样本数过少、纯度足够
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            # 叶节点标签取节点内多数类（样本数最多的类别）
            return {'leaf': True, 'label': Counter(y).most_common(1)[0][0]}

        # 寻找最佳分裂特征和阈值
        best_feature, best_threshold, best_gini = None, None, float('inf')

        for feature_idx in range(n_features):
            # 候选阈值取该特征的所有不同取值（连续特征按取值切分，x_j <= t 与 x_j > t）
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
        # 按 (feature_idx, threshold) 计算分裂后的加权基尼不纯度
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # 任一侧为空集时该分裂无意义，返回 inf 使其不会被选中
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return float('inf')

        # 基尼系数 Gini(S) = 1 - Σ_k p_k^2，p_k 为子集 S 中第 k 类的占比
        gini_left = 1 - np.sum([(np.sum(y[left_mask] == c) / np.sum(left_mask)) ** 2
                                for c in np.unique(y)])
        gini_right = 1 - np.sum([(np.sum(y[right_mask] == c) / np.sum(right_mask)) ** 2
                                 for c in np.unique(y)])

        n = len(y)
        # 加权基尼 = (|L|/n)*Gini(L) + (|R|/n)*Gini(R)，即分裂后的期望不纯度
        gini = (np.sum(left_mask) / n) * gini_left + (np.sum(right_mask) / n) * gini_right  # # 加权基尼系数
        return gini

    def predict(self, X):
        # 逐样本从根节点走到底
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        # 递归下行：满足 x_j <= t 走左子树，否则走右子树
        if tree['leaf']:
            return tree['label']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])


class RandomForest:
    """随机森林：多棵 CART 树投票。"""

    """
    随机森林分类器
    通过构建多棵决策树并投票来提高泛化能力
    """

    def __init__(self, n_trees=100, max_depth=5, min_samples_split=2, max_features=None):
        # n_trees：森林中决策树的数量
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 每棵树使用的特征数，默认 sqrt(n_features)
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        # 默认使用 sqrt(n_features) 个特征
        # 分类任务经验最优（Breiman 2001）：既保持单棵树有一定能力，又保证树间多样性
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_trees):
            # Bootstrap 采样：有放回地随机抽取样本
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 随机选择特征子集
            # 每棵树只用无放回抽出的 max_features 个特征建树——随机森林区别于普通 Bagging 的关键
            # 该机制去除了树间相关性，从而降低整体方差
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_bootstrap[:, feature_indices]

            # 训练决策树
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_bootstrap)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # 每棵树投票
        # 每棵树只接收自己训练时使用的特征子集 X[:, features]
        predictions = np.array([tree.predict(X[:, features])
                                for tree, features in self.trees])
        # 多数投票
        # 对每个样本（predictions 的一列）统计各类得票，取最多的类别
        return np.array([Counter(col).most_common(1)[0][0]
                         for col in predictions.T])

    def score(self, X, y):
        # 准确率 = 预测正确比例
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
