# -*- coding: utf-8 -*-
"""PCA 单元测试。"""
import numpy as np
import pytest
from PCA.pca import pca, zeroMean, percent2n


class TestZeroMean:
    """zeroMean 函数测试。"""

    def test_output_shape(self):
        """中心化后形状不变。"""
        X = np.random.randn(50, 5)
        centered, mean = zeroMean(X)
        assert centered.shape == X.shape
        assert mean.shape == (5,)

    def test_mean_zero(self):
        """中心化后每个特征均值为 0。"""
        X = np.random.randn(100, 4)
        centered, _ = zeroMean(X)
        assert np.allclose(centered.mean(axis=0), 0, atol=1e-12)


class TestPercent2n:
    """percent2n 函数测试。"""

    def test_all_components(self):
        """100% 贡献率保留全部特征值。"""
        eigVals = np.array([5.0, 3.0, 2.0, 1.0])
        assert percent2n(eigVals, 1.0) == 4

    def test_subset(self):
        """50% 贡献率保留部分特征值。"""
        eigVals = np.array([8.0, 1.0, 0.5, 0.5])
        n = percent2n(eigVals, 0.5)
        assert 1 <= n <= 4


class TestPCA:
    """pca 函数测试。"""

    def test_output_shapes(self):
        """降维数据形状为 (n, k)，重构数据形状等于原始。"""
        # 构造低秩数据，保证 PCA 能降维
        rng = np.random.RandomState(42)
        X = rng.randn(80, 1) @ rng.randn(1, 6) + rng.randn(80, 6) * 0.1
        low, rec = pca(X, percent=0.9)
        assert low.shape[0] == 80
        assert low.shape[1] < 6  # 低秩数据应显著降维
        assert rec.shape == X.shape

    def test_full_reconstruction(self):
        """100% 贡献率时重构应接近原始。"""
        X = np.random.RandomState(42).randn(60, 3)
        low, rec = pca(X, percent=1.0)
        assert np.allclose(rec, X, atol=1e-8)
