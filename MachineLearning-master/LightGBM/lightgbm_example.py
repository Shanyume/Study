"""LightGBM 分类示例：Iris 数据集。"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbose=-1)
    model.fit(Xtr, ytr)
    print("test acc:", accuracy_score(yte, model.predict(Xte)))
    print("feature importance:", model.feature_importances_)

if __name__ == "__main__":
    main()
