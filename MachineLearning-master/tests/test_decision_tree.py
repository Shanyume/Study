# -*- coding: utf-8 -*-
"""决策树单元测试。"""
import numpy as np
import pytest
from DecisionTree.id3_c45 import DecisionTree, NotFittedError


@pytest.fixture
def toy_data():
    """手工小数据集。"""
    X = [[1, 2, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [1, 0, 0, 0, 1],
         [2, 1, 1, 0, 1],
         [1, 1, 0, 1, 1]]
    y = ['yes', 'yes', 'no', 'no', 'no']
    return np.array(X), np.array(y)


class TestDecisionTree:
    """决策树测试。"""

    def test_id3_fit_predict(self, toy_data):
        """ID3 训练集预测应等于标签。"""
        X, y = toy_data
        clf = DecisionTree(mode='ID3').fit(X, y)
        assert list(clf.predict(X)) == list(y)

    def test_c45_fit_predict(self, toy_data):
        """C4.5 训练集预测应等于标签。"""
        X, y = toy_data
        clf = DecisionTree(mode='C4.5').fit(X, y)
        assert list(clf.predict(X)) == list(y)

    def test_invalid_mode(self):
        """无效 mode 应抛异常。"""
        with pytest.raises(Exception):
            DecisionTree(mode='invalid')

    def test_not_fitted_error(self, toy_data):
        """未训练就 predict 应抛 NotFittedError。"""
        X, y = toy_data
        clf = DecisionTree()
        with pytest.raises(NotFittedError):
            clf.predict(X)
