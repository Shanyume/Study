# -*- coding: utf-8 -*-
"""
SVM by Quadratic Programming（二次规划 SVM）
==============================================

用 cvxopt 求解 SVM 对偶问题的二次规划：
    - 构建拉格朗日乘子 alpha 的 QP 问题
    - 支持线性核、多项式核、高斯核
    - 硬间隔（C=None）和软间隔（C 设定值）

依赖：numpy、cvxopt
"""
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

## define kenrel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
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
        n_samples, n_features = X.shape

        # Gram matrix
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
        return np.sign(self.project(X))

