"""通用模型评估脚本：ROC 曲线、PR 曲线、混淆矩阵。"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, RocCurveDisplay
from pathlib import Path

BASE = Path(__file__).resolve().parent

def main():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        model.fit(Xtr, ytr)
        scores = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, scores)
        print(f"{name}: ROC-AUC={auc(fpr, tpr):.4f}")
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=0.4); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend()
    plt.subplot(1, 2, 2)
    for name, model in models.items():
        scores = model.predict_proba(Xte)[:, 1]
        prec, rec, _ = precision_recall_curve(yte, scores)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / "roc_pr_curves.png", dpi=120)
    print("figure saved:", BASE / "roc_pr_curves.png")

if __name__ == "__main__":
    main()
