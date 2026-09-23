"""CART 决策树独立实现（基尼系数，二叉分裂）。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CARTClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth; self.min_split = min_samples_split
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)
    def _best_split(self, X, y):
        best = (None, None, np.inf)
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left, right = y[X[:, j] <= t], y[X[:, j] > t]
                if len(left) < self.min_split or len(right) < self.min_split: continue
                g = (len(left)*self._gini(left) + len(right)*self._gini(right)) / len(y)
                if g < best[2]: best = (j, t, g)
        return best
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "label": np.bincount(y).argmax()}
        j, t, g = self._best_split(X, y)
        if j is None: return {"leaf": True, "label": np.bincount(y).argmax()}
        left = self._build(X[X[:, j] <= t], y[X[:, j] <= t], depth+1)
        right = self._build(X[X[:, j] > t], y[X[:, j] > t], depth+1)
        return {"leaf": False, "j": j, "t": t, "left": left, "right": right}
    def fit(self, X, y):
        self.tree = self._build(np.asarray(X), np.asarray(y), 0); return self
    def _predict_one(self, x, node):
        while not node["leaf"]:
            node = node["left"] if x[node["j"]] <= node["t"] else node["right"]
        return node["label"]
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = CARTClassifier(max_depth=4).fit(Xtr, ytr)
    print("CART test acc:", accuracy_score(yte, model.predict(Xte)))
