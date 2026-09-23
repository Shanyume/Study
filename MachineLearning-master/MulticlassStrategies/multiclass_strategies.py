"""One-vs-All 与 One-vs-One 多分类策略演示（基于 sklearn LogisticRegression）。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    ova = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    print("OvA test acc:", ova.score(Xte, yte))
    print("OvO test acc:", ovo.score(Xte, yte))

if __name__ == "__main__":
    main()
