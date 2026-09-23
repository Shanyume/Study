"""Softmax 回归（多分类逻辑回归）NumPy 实现，Iris 数据集。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SoftmaxRegression:
    def __init__(self, lr=0.1, n_iters=500, l2=1e-4):
        self.lr = lr; self.n_iters = n_iters; self.l2 = l2
    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        n, d = X.shape; self.k = len(np.unique(y))
        Y = np.eye(self.k)[y]
        self.W = np.zeros((d, self.k)); self.b = np.zeros(self.k)
        for _ in range(self.n_iters):
            P = self._softmax(X @ self.W + self.b)
            grad_w = X.T @ (P - Y) / n + self.l2 * self.W
            grad_b = (P - Y).mean(axis=0)
            self.W -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self
    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    model = SoftmaxRegression(lr=0.2, n_iters=1000).fit(scaler.transform(Xtr), ytr)
    print("Softmax test acc:", accuracy_score(yte, model.predict(scaler.transform(Xte))))
