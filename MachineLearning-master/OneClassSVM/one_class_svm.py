"""One-Class SVM 异常检测示例。"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def main():
    rng = np.random.RandomState(42)
    X_normal = rng.randn(300, 2)
    X_outliers = rng.uniform(low=-8, high=8, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])
    X = StandardScaler().fit_transform(X)
    model = OneClassSVM(kernel="rbf", nu=0.06, gamma="scale")
    labels = model.fit_predict(X)
    print("detected outliers:", np.sum(labels == -1), "/", len(X))
    print("expected outliers:  20")

if __name__ == "__main__":
    main()
