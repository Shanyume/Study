"""Gradient Boosting 二分类简化实现（决策树桩 + 梯度提升）。"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleTreeStump:
    """深度为 1 的回归树桩，用于拟合残差。"""
    def fit(self, X, y):
        best = (None, None, np.inf)
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left = y[X[:, j] <= t]; right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0: continue
                err = ((left - left.mean())**2).sum() + ((right - right.mean())**2).sum()
                if err < best[2]: best = (j, t, err)
        self.j, self.t = best[0], best[1]
        self.left = y[X[:, self.j] <= self.t].mean() if self.j is not None else 0.0
        self.right = y[X[:, self.j] > self.t].mean() if self.j is not None else 0.0
        return self
    def predict(self, X):
        return np.where(X[:, self.j] <= self.t, self.left, self.right)

class GradientBoostingClassifier:
    """学习率 learning_rate，迭代 n_estimators 次拟合负梯度（logistic loss）。"""
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators; self.lr = learning_rate; self.trees = []
    def fit(self, X, y):
        y = np.where(y == 1, 1.0, -1.0)
        F = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residual = y / (1 + np.exp(y * F))  # logistic loss 负梯度
            tree = SimpleTreeStump().fit(X, residual)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self
    def decision_function(self, X):
        return sum(self.lr * t.predict(X) for t in self.trees)
    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

if __name__ == "__main__":
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.2).fit(Xtr, ytr)
    print("train acc:", accuracy_score(ytr, model.predict(Xtr)))
    print("test  acc:", accuracy_score(yte, model.predict(Xte)))
