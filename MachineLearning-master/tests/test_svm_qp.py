# -*- coding: utf-8 -*-
"""SVM QP 单元测试。"""
import numpy as np
import pytest
from SVM.SVM_by_QP.SVCQP import SVM, linear_kernel, gaussian_kernel, polynomial_kernel


class TestKernelFunctions:
    """核函数测试。"""

    def test_linear_kernel(self):
        """线性核 = 点积。"""
        x1 = np.array([1, 2]); x2 = np.array([3, 4])
        assert np.isclose(linear_kernel(x1, x2), 1*3 + 2*4)

    def test_gaussian_kernel_range(self):
        """高斯核取值在 (0, 1]。"""
        x1 = np.array([0, 0]); x2 = np.array([1, 1])
        k = gaussian_kernel(x1, x2, sigma=1.0)
        assert 0 < k <= 1

    def test_polynomial_kernel(self):
        """多项式核公式。"""
        x1 = np.array([1, 2]); x2 = np.array([3, 4])
        p = 3
        expected = (1 + (1*3 + 2*4)) ** p
        assert np.isclose(polynomial_kernel(x1, x2, p=p), expected)


class TestSVMQP:
    """QP SVM 测试。"""

    def test_linear_separable(self):
        """线性可分小样本，预测应等于标签。"""
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([-1, -1, 1, 1])
        clf = SVM(kernel=linear_kernel, C=1.0)
        clf.fit(X, y)
        assert np.allclose(clf.predict(X), y)
