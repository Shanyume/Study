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

# ============================================================
# 数学推导与算法要点
# ============================================================
# 1. 贝叶斯定理：
#        P(y=c_k | x) = P(x | y=c_k) * P(y=c_k) / P(x)
#    分母 P(x) 与类别无关，比较各类后验大小时可省略，故
#    预测规则为 y = argmax_k [ P(y=c_k) * Π_j P(x_j | y=c_k) ]。
#
# 2. "朴素"假设：给定类别时各特征条件独立，即
#        P(x_1, ..., x_n | y=c_k) = Π_j P(x_j | y=c_k)
#    该假设一般不成立，但因只关心 argmax，实践中效果往往很好。
#
# 3. 概率估计（极大似然 / 频率估计）：
#        P(y=c_k)        ≈ 该类别样本数 / 总样本数
#        P(x_j = v|y=c_k) ≈ 类别 c_k 中特征 j 取值 v 的次数
#                           / 类别 c_k 样本数
#    零概率问题：训练时未出现过的特征值 v，其概率估为 0，
#    乘积中含一个 0 会使整个后验为 0。平滑（加伪计数）解决之：
#        P(x_j=v|c_k) = (count(v|c_k) + alpha)
#                      / (N_c_k + alpha * V_j)
#    其中 V_j 为特征 j 的取值个数；alpha=1 即 Laplace 平滑。
#
# 4. 数值稳定性：多个 (0,1) 小数连乘极易下溢为 0（float64
#    最小正规数约 1e-308），工程上通常在对数空间累加：
#        log P(y=c_k|x) = log P(y=c_k) + Σ_j log P(x_j|c_k)
#    本实现按"简单演示"保留连乘形式（见 _predict_single_sample）。
#
# 5. 高斯模型：对连续特征假设 x_j | y=c_k ~ N(mu_jk, sigma_jk^2)，
#        P(x_j|c_k) = 1/(sigma*sqrt(2*pi)) * exp(-(x_j-mu)^2/(2*sigma^2))
#    参数用极大似然估计：mu = 样本均值，sigma^2 = 样本方差。
# ============================================================

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
        # 三个超参数的作用小结：
        #   alpha       —— 控制平滑强度：越大，估计越趋向"平均分布"，
        #                  方差越小、偏差越大（偏差-方差权衡）；
        #   fit_prior   —— 先验用频率估计还是均匀分布 1/K；
        #   class_prior —— 直接指定先验（此时忽略 fit_prior）。

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
        # 平滑公式含义：分子 = 真实计数 + 伪计数 alpha，
        # 分母 = 样本总数 + alpha * V_j（V_j 为取值个数），
        # 保证所有概率之和恰为 1，且任何取值概率都不为 0
        #（alpha>0 时最小概率 = alpha/(N+alpha*V_j)）。
        # 极限情况：alpha -> 0 退化为频率估计（极大似然）。
        return value_prob

    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器。
        本模型是"生成式"模型：训练时估计每个类别下的参数
            （先验 P(y=c_k) + 每个特征的条件分布 P(x_j|y=c_k)），
        预测时再按贝叶斯公式合成后验。这与 logistic 回归等
        "判别式"模型（直接建模 P(y|x)）思路不同；生成式模型的
        优势是数据效率高、能自然处理缺失特征，且可复用于
        密度估计等任务。
        
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
        # 先验 = 类别 c 的样本数占比（MLE 估计）；这里套用与
        # 条件概率相同的平滑公式 (N_c + alpha)/(N + alpha*K)，
        # 保证小样本下先验也不出现 0，且各类先验之和为 1。
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
        计算规则（MAP 估计，最大后验）：
            y = argmax_c [ P(y=c) * Π_j P(x_j | y=c) ]
        分母 P(x) 对所有类别相同，argmax 时省略。
        注意：这里用的是连乘而非 log 求和，见文件头部
        "数值稳定性"一节——特征多或概率小时会下溢为 0，
        届时多个类别同判 0，结果将取决于先验大小。
        
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
    模型形式：
        P(y=c_k|x) ∝ P(y=c_k) * Π_j N(x_j; mu_jk, sigma_jk^2)
    训练即极大似然估计：mu_jk 取类别 c_k 下特征 j 的样本均值，
    sigma_jk 取样本标准差；预测仍是 argmax 最大后验（MAP）。
    与多项式版本相比，高斯版无需对特征离散化，天然适合
    连续/数值型特征，但要求"每类内特征近似正态"。
    
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
        # 说明：MLE 的方差估计分母是 N（而非 N-1），略向下偏；
        # 小样本下可用 N-1（无偏估计）改善，此处从简。
        # sigma 兜底 1e-9：若某类下特征恒定（方差为 0），
        # 高斯密度会退化为 delta 函数导致除零/inf，加 epsilon
        # 使其退化为极窄的高斯，数值上可安全计算。
        return (mu, sigma)

    def _prob_gaussian(self, mu, sigma, x):
        """
        高斯概率密度函数：
            p(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)^2 / (2*sigma^2))
        注意：连续特征取的是概率密度而非概率值（可为 >1），
        但因比较的只是 argmax 后验，常数因子不影响结果，
        所以不需要像离散分布那样归一化。
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
                # （与父类 MultinomialNB 的区别：父类先验带 alpha
                #  平滑，这里就是简单的 N_c/N；类别数通常不多，
                #  先验为 0 的概率很低，不影响数值稳定）
                self.class_prior_ = []
                for c in self.classes_:
                    c_num = np.sum(np.equal(y, c))
                    self.class_prior_.append(c_num / n_samples)
                self.class_prior_ = np.array(self.class_prior_)

        # ---------- 计算条件概率（高斯参数） ----------
        # 这里"条件概率"实际存的是 (mu, sigma) 参数元组，
        # 由 predict 路径中的 _get_xj_prob（已重写）在预测时
        # 即时代入高斯密度公式求值，而非存储离散的取值概率。
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