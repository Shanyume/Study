# -*- coding: utf-8 -*-
"""SVM SMO 单元测试。"""
import numpy as np
import csv
import pytest
from SVM.SVM_by_SMO.SVCSMO import SVCSMO


@pytest.fixture
def iris_binary():
    """加载 Iris 二分类数据。"""
    data_file = __import__('pathlib').Path(__file__).parent.parent / 'SVM' / 'SVM_by_SMO' / 'small_data' / 'iris-slwc.txt'
    data = []
    with open(data_file) as f:
        for row in csv.reader(f, delimiter=','):
            data.append([float(x) for x in row])
    data = np.array(data)
    return data[:, :-1], data[:, -1].astype(int)


class TestSVCSMO:
    """SMO SVM 测试。"""

    def test_linear_kernel_accuracy(self, iris_binary):
        """线性核训练集准确率应 > 0.8。"""
        X, y = iris_binary
        model = SVCSMO(max_iter=100, kernel_type='linear')
        model.fit(X, y)
        y_hat = model.predict(X)
        acc = np.mean(y_hat == y)
        assert acc > 0.8

    def test_returns_support_vectors(self, iris_binary):
        """fit 应返回支持向量数组和迭代次数。"""
        X, y = iris_binary
        model = SVCSMO(max_iter=50)
        result = model.fit(X, y)
        assert isinstance(result, tuple) and len(result) == 2
        support_vectors, iterations = result
        assert support_vectors.shape[1] == X.shape[1]
        assert iterations > 0
