# -*- coding: utf-8 -*-
"""KMeans 单元测试。"""
import numpy as np
import pickle
import pytest
from KMeans.kmeans import KMeans


@pytest.fixture
def iris_data():
    """加载 KMeans 测试数据。"""
    data_file = __import__('pathlib').Path(__file__).parent.parent / 'KMeans' / 'data.pkl'
    with open(data_file, 'rb') as f:
        X, y = pickle.load(f, encoding='latin1')
    return X, y


class TestKMeans:
    """KMeans 聚类测试。"""

    def test_output_shapes(self, iris_data):
        """标签数量等于样本数，质心数量等于 k。"""
        X, y = iris_data
        km = KMeans(n_clusters=3, initCent=X[50:53], max_iter=20)
        km.fit(X)
        assert km.labels.shape == (len(X),)
        assert km.centroids.shape == (3, X.shape[1])

    def test_sse_positive(self, iris_data):
        """SSE 应为正数。"""
        X, y = iris_data
        km = KMeans(n_clusters=3, initCent=X[50:53], max_iter=20)
        km.fit(X)
        assert km.sse > 0

    def test_predict_new_data(self, iris_data):
        """predict 返回正确的样本数。"""
        X, y = iris_data
        km = KMeans(n_clusters=3, initCent=X[50:53], max_iter=20)
        km.fit(X)
        X_new = X[:10]
        preds = km.predict(X_new)
        assert preds.shape == (10,)
        assert np.all((preds >= 0) & (preds < 3))
