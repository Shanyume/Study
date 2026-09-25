# -*- coding: utf-8 -*-
"""Bagging 单元测试。"""
import numpy as np
import pytest
from Bagging.bagging import BaggingClassifier


class TestBagging:
    """Bagging 测试。"""

    def test_fit_predict_shape(self):
        """预测数量等于样本数。"""
        X = np.random.RandomState(42).randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        model = BaggingClassifier(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_n_estimators(self):
        """模型数量等于 n_estimators。"""
        X = np.random.RandomState(42).randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        model = BaggingClassifier(n_estimators=7)
        model.fit(X, y)
        assert len(model.models) == 7
