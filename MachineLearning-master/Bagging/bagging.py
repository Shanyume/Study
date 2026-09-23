"""Bagging 独立实现：自助采样 + 多个决策树投票。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest.random_forest import DecisionTree

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=4, random_state=42):
        self.n = n_estimators; self.max_depth = max_depth; self.seed = random_state
    def fit(self, X, y):
        rng = np.random.RandomState(self.seed); self.models = []
        for i in range(self.n):
            idx = rng.randint(0, len(X), len(X))
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[idx], y[idx]); self.models.append(tree)
        return self
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.array([np.bincount(p.astype(int)).argmax() for p in preds.T])

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = BaggingClassifier(n_estimators=15).fit(Xtr, ytr)
    print("bagging test acc:", accuracy_score(yte, model.predict(Xte)))
