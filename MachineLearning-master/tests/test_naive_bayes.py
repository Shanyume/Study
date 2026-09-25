# -*- coding: utf-8 -*-
"""朴素贝叶斯单元测试。"""
import numpy as np
import pytest
from NaiveBayes.NaiveBayes import MultinomialNB, GaussianNB


@pytest.fixture
def discrete_data():
    """离散特征数据。"""
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ]).T
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, y


class TestMultinomialNB:
    """多项式朴素贝叶斯测试。"""

    def test_fit(self, discrete_data):
        """训练后 classes_ 和条件概率存在。"""
        X, y = discrete_data
        nb = MultinomialNB(alpha=1.0, fit_prior=True)
        nb.fit(X, y)
        assert nb.classes_ is not None
        assert nb.conditional_prob_ is not None
        assert len(nb.classes_) == 2

    def test_predict_valid_class(self, discrete_data):
        """预测类别应在训练类别中。"""
        X, y = discrete_data
        nb = MultinomialNB(alpha=1.0).fit(X, y)
        pred = nb.predict(np.array([2, 4]))
        assert pred in nb.classes_

    def test_laplace_smoothing(self, discrete_data):
        """Laplace 平滑（alpha=1）条件概率应都不为 0。"""
        X, y = discrete_data
        nb = MultinomialNB(alpha=1.0).fit(X, y)
        for c in nb.classes_:
            for feat_idx in nb.conditional_prob_[c]:
                for v, prob in nb.conditional_prob_[c][feat_idx].items():
                    assert prob > 0


class TestGaussianNB:
    """高斯朴素贝叶斯测试。"""

    def test_fit_predict(self, discrete_data):
        """高斯朴素贝叶斯训练集预测数量等于样本数。"""
        X, y = discrete_data
        gnb = GaussianNB()
        preds = gnb.fit(X, y).predict(X)
        assert len(preds) == len(X)
