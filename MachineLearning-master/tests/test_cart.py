# -*- coding: utf-8 -*-
"""CART 单元测试。"""
import numpy as np
import pytest
from CART.cart import CARTClassifier


class TestCART:
    """CART 决策树测试。"""

    def test_fit_predict_shape(self):
        """预测数量等于样本数。"""
        X = np.random.RandomState(42).randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        model = CARTClassifier(max_depth=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_pure_data(self):
        """纯数据应直接成为叶子。"""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 1])
        model = CARTClassifier(max_depth=5)
        model.fit(X, y)
        assert model.tree.get('leaf', False)
