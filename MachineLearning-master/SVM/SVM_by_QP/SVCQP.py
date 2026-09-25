# -*- coding: utf-8 -*-
"""
SVM by Quadratic Programming（二次规划 SVM）
==============================================

SVM 的核心是求解对偶问题的二次规划（QP）：

    max  Σ alpha_i - (1/2) Σ Σ alpha_i alpha_j y_i y_j K(x_i, x_j)
    s.t. Σ alpha_i y_i = 0
         0 <= alpha_i <= C    （软间隔）
         alpha_i >= 0          （硬间隔）

cvxopt 用标准形式：
    min  (1/2) alpha^T P alpha + q^T alpha
    s.t. G alpha <= h
         A alpha = b

其中：
    P = outer(y, y) * K     （核矩阵按标签加权）
    q = -1（全 -1 向量）
    A = y^T                 （等式约束 y^T alpha = 0）
    b = 0
    G/h：约束 alpha 的下界和上界

支持核函数：
    - linear_kernel：线性核
    - polynomial_kernel：多项式核
    - gaussian_kernel：高斯核（RBF）

依赖：numpy、cvxopt
"""
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

## 定义核函数
def linear_kernel(x1, x2):
    """线性核：K(x, y) = x · y。"""
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    """多项式核：K(x, y) = (1 + x·y)^p，p 是次数。"""
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    """高斯核（RBF）：K(x, y) = exp(-||x-y||² / (2σ²))，σ 控制影响范围。"""
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
## end define kernel functions

class SVM(object):
    """
    Suppoet vector classification by quadratic programming
    """

    def __init__(self, kernel=linear_kernel, C=None):
        """

        :param kernel: kernel types, should be in the kernel function list above
        :param C:
        """
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        """
        训练 SVM：构建 QP 问题并求解。

        :param X: 训练特征矩阵 (n_samples, n_features)
        :param y: 训练标签 (+1 / -1)
        """
        n_samples, n_features = X.shape

        # 构建核矩阵（Gram matrix）：K[i,j] = kernel(X[i], X[j])
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)  # # QP 目标函数 P 矩阵：(y*y)^T * K
        q = cvxopt.matrix(np.ones(n_samples) * -1)  # # QP 目标函数 q 向量：全 -1
        A = cvxopt.matrix(y.astype(float), (1,n_samples))  # # 等式约束：y^T * alpha = 0
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))  # # 硬间隔约束：alpha >= 0
            h = cvxopt.matrix(np.zeros(n_samples))  # # 约束右侧：全 0
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem, DOC: http://cvxopt.org/userguide/coneprog.html?highlight=qp#cvxopt.solvers.qp
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)  # # 求解 QP 问题

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5  # # 支持向量：alpha > 0 的样本
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]  # # w = Σ alpha_i * y_i * x_i
        else:
            self.w = None

    def project(self, X):
        """计算决策函数值 w·x + b（线性核）或 Σ alpha_i*y_i*K(x,sv)+b（非线性核）。"""
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):  # # 累加所有支持向量的加权核输出
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        """预测：决策函数值 > 0 为 +1，< 0 为 -1。"""
        return np.sign(self.project(X))

