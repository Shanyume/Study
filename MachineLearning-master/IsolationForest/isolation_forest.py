"""Isolation Forest 异常检测示例。"""
import numpy as np
from sklearn.ensemble import IsolationForest

def main():
    # 正常数据 + 少量异常点
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])
    model = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
    labels = model.fit_predict(X)  # -1 表示异常
    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")

if __name__ == "__main__":
    main()
