"""XGBoost 分类示例：Iris 数据集 + 特征重要性。"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss")
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("test acc:", accuracy_score(yte, pred))
    print(classification_report(yte, pred))
    print("feature importance:", model.feature_importances_)

if __name__ == "__main__":
    main()
