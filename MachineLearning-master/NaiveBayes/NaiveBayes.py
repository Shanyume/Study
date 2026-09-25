# -*- coding: utf-8 -*-
"""
Naive Bayes（朴素贝叶斯）
==========================

朴素贝叶斯基于贝叶斯定理和特征条件独立假设：

    P(y = c_k | x) ∝ P(y = c_k) * Π_j P(x_j | y = c_k)

预测时选择使后验概率最大的类别。

提供两个类：
    - MultinomialNB：多项式朴素贝叶斯，适用于离散特征（如文本词频）
        支持 Laplace 平滑（alpha=1）和 Lidstone 平滑（0<alpha<1）
    - GaussianNB：高斯朴素贝叶斯，适用于连续特征
        假设每个特征在每个类别下服从高斯分布

API 参考：http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes

依赖：numpy
"""
import numpy as np

class MultinomialNB(object):
    """
    多项式朴素贝叶斯分类器，适用于离散特征（如文本分类中的词频）。
    
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        平滑参数：
        - alpha = 0 表示无平滑
        - 0 < alpha < 1 为 Lidstone 平滑
        - alpha = 1 为 Laplace 平滑
    fit_prior : bool, optional (default=True)
        是否从数据中学习类先验概率。若为 False，则使用均匀先验。
    class_prior : array-like, size (n_classes,), optional
        类别的先验概率，若指定则直接使用，不再根据数据调整。
    
    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        类别标签数组。
    class_prior_ : ndarray, shape (n_classes,)
        每个类别的先验概率。
    conditional_prob_ : dict
        条件概率字典，结构为 {类别: {特征索引: {特征值: 概率}}}
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = None
        self.conditional_prob_ = None

    def _calculate_feature_prob(self, feature):
        """
        计算单个特征在某个类别下的概率分布（带平滑）。
        
        Parameters
        ----------
        feature : ndarray, shape (n_samples,)
            该类别的样本在该特征上的取值。
        
        Returns
        -------
        value_prob : dict
            键为特征值，值为对应的概率 P(特征值 | 类别)。
        """
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            # 计算出现次数，加上平滑项
            count = np.sum(np.equal(feature, v))
            prob = (count + self.alpha) / (total_num + len(values) * self.alpha)
            value_prob[v] = prob
        return value_prob

    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器。
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            训练样本特征矩阵。
        y : array-like, shape (n_samples,)
            训练样本对应的类别标签。
        
        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        n_features = X.shape[1]

        # ---------- 计算类先验概率 P(y=ck) ----------
        if self.class_prior is not None:
            # 如果用户指定了先验，直接使用（需检查长度）
            if len(self.class_prior) != n_classes:
                raise ValueError("class_prior length must match number of classes")
            self.class_prior_ = np.asarray(self.class_prior)
        else:
            if not self.fit_prior:
                # 均匀先验
                self.class_prior_ = np.full(n_classes, 1.0 / n_classes)
            else:
                # 从数据中计算，并应用同样的平滑（保持与条件概率平滑一致）
                self.class_prior_ = []
                for c in self.classes_:
                    c_num = np.sum(np.equal(y, c))
                    # 使用与条件概率相同的 alpha 进行平滑
                    prior = (c_num + self.alpha) / (n_samples + n_classes * self.alpha)
                    self.class_prior_.append(prior)
                self.class_prior_ = np.array(self.class_prior_)

        # ---------- 计算条件概率 P(xj | y=ck) ----------
        self.conditional_prob_ = {}  # 字典：类别 -> 特征索引 -> 特征值概率字典
        for c in self.classes_:
            # 取出属于类别 c 的所有样本
            mask = (y == c)
            X_c = X[mask]
            # 初始化该类别的条件概率字典
            self.conditional_prob_[c] = {}
            for i in range(n_features):
                feature_i = X_c[:, i]
                # 计算该特征在类别 c 下的概率分布
                self.conditional_prob_[c][i] = self._calculate_feature_prob(feature_i)
        return self

    def _get_xj_prob(self, values_prob, target_value):
        """
        获取某个特征值在给定概率分布下的概率。
        
        Parameters
        ----------
        values_prob : dict
            特征值到概率的映射。
        target_value : 
            待查询的特征值。
        
        Returns
        -------
        prob : float
            对应的概率；若特征值不在字典中，返回 0（避免后续乘法出错）。
        """
        return values_prob.get(target_value, 0.0)

    def _predict_single_sample(self, x):
        """
        对单个样本进行预测。
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            待预测样本的特征向量。
        
        Returns
        -------
        label : 
            预测的类别标签。
        """
        max_posterior = -np.inf
        best_label = None

        # 遍历每个类别，计算后验概率
        for idx, c in enumerate(self.classes_):
            # 该类别的先验
            prior = self.class_prior_[idx]
            # 计算条件概率乘积（对数空间下改为求和，防止下溢）
            # 为数值稳定性，使用对数，但这里保持原乘法（简单演示）
            # 若特征值缺失，概率为0，则后验为0，可跳过
            cond_prob = 1.0
            for feat_idx, feat_val in enumerate(x):  # # 遍历每个特征
                feat_prob_dict = self.conditional_prob_[c][feat_idx]
                prob = self._get_xj_prob(feat_prob_dict, feat_val)
                cond_prob *= prob
                # 若某特征概率为0，则整体为0，可提前跳出
                if cond_prob == 0:
                    break
            posterior = prior * cond_prob
            if posterior > max_posterior:
                max_posterior = posterior
                best_label = c
        return best_label

    def predict(self, X):
        """
        对样本（单个或多个）进行预测。
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) 或 (n_features,)
            待预测样本。
        
        Returns
        -------
        labels : ndarray, shape (n_samples,) 或 scalar
            预测的类别标签。
        """
        X = np.asarray(X)
        if self.classes_ is None or self.conditional_prob_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        if X.ndim == 1:
            # 单个样本
            return self._predict_single_sample(X)
        else:
            # 多个样本
            labels = [self._predict_single_sample(x) for x in X]
            return np.array(labels)


class GaussianNB(MultinomialNB):
    """
    高斯朴素贝叶斯分类器，适用于连续特征，假设特征服从高斯分布。
    继承自 MultinomialNB，但重写了条件概率的计算方法。
    
    注意：由于继承，父类的 alpha 和 fit_prior 参数依然存在，
    但在高斯模型中，alpha 仅用于先验平滑（若 fit_prior=True），
    对条件概率无影响。
    """
    def _calculate_feature_prob(self, feature):
        """
        计算单个特征在某个类别下的高斯分布参数（均值和标准差）。
        
        Parameters
        ----------
        feature : ndarray, shape (n_samples,)
            该类别的样本在该特征上的取值。
        
        Returns
        -------
        (mu, sigma) : tuple
            均值和标准差。
        """
        mu = np.mean(feature)
        sigma = np.std(feature)
        # 防止方差为0，添加极小值
        if sigma == 0:
            sigma = 1e-9
        return (mu, sigma)

    def _prob_gaussian(self, mu, sigma, x):
        """
        高斯概率密度函数。
        """
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def _get_xj_prob(self, mu_sigma, target_value):
        """
        获取连续特征值在高斯分布下的概率密度值。
        """
        mu, sigma = mu_sigma
        return self._prob_gaussian(mu, sigma, target_value)

    def fit(self, X, y):
        """
        训练高斯朴素贝叶斯分类器。
        重写以调整先验计算（不使用 alpha 平滑，除非用户指定 class_prior）。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        n_features = X.shape[1]

        # ---------- 计算类先验概率 ----------
        if self.class_prior is not None:
            if len(self.class_prior) != n_classes:
                raise ValueError("class_prior length must match number of classes")
            self.class_prior_ = np.asarray(self.class_prior)
        else:
            if not self.fit_prior:
                self.class_prior_ = np.full(n_classes, 1.0 / n_classes)
            else:
                # 高斯先验通常不使用平滑，直接用频率
                self.class_prior_ = []
                for c in self.classes_:
                    c_num = np.sum(np.equal(y, c))
                    self.class_prior_.append(c_num / n_samples)
                self.class_prior_ = np.array(self.class_prior_)

        # ---------- 计算条件概率（高斯参数） ----------
        self.conditional_prob_ = {}
        for c in self.classes_:
            mask = (y == c)
            X_c = X[mask]
            self.conditional_prob_[c] = {}
            for i in range(n_features):
                feature_i = X_c[:, i]
                # 计算均值和标准差
                self.conditional_prob_[c][i] = self._calculate_feature_prob(feature_i)
        return self


# 简单测试（可删除）
if __name__ == "__main__":
    # 测试多项式朴素贝叶斯
    X = np.array([[1, 2], [1, 1], [2, 2], [2, 1]])
    y = np.array([0, 0, 1, 1])
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X, y)
    print(mnb.predict(np.array([1, 2])))  # 应输出 0

    # 测试高斯朴素贝叶斯
    Xg = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    yg = np.array([0, 0, 0, 1, 1, 1])
    gnb = GaussianNB()
    gnb.fit(Xg, yg)
    print(gnb.predict(np.array([[-0.8, -1]])))  # 应输出 0