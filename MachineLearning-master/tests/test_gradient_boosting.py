# -*- coding: utf-8 -*-
"""Gradient Boosting 单元测试。"""
import numpy as np
import pytest
from Boosting.GradientBoosting.gradient_boosting import GradientBoostingClassifier, SimpleTreeStump


class TestSimpleTreeStump:
    """树桩测试。"""

    def test_fit_predict(self):
        """树桩预测应只输出两个值（左右叶）。"""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([-1, -1, 1, 1])
        stump = SimpleTreeStump().fit(X, y)
        preds = stump.predict(X)
        assert len(set(preds)) <= 2


class TestGradientBoosting:
    """梯度提升测试。"""

    def test_fit_predict_shape(self):
        """预测数量等于样本数。"""
        X = np.random.RandomState(42).randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.2)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_learning(self):
        """更多迭代轮数应提高训练集准确率。"""
        X = np.random.RandomState(42).randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        acc1 = np.mean(GradientBoostingClassifier(n_estimators=5).fit(X, y).predict(X) == y)
        acc2 = np.mean(GradientBoostingClassifier(n_estimators=50).fit(X, y).predict(X) == y)
        assert acc2 >= acc1
