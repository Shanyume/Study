#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数搜索：GridSearchCV 与 RandomizedSearchCV
================================================

- GridSearchCV：穷举网格中所有参数组合，适合参数空间小
- RandomizedSearchCV：随机采样参数组合，适合参数空间大

数据集：sklearn Iris
依赖：scikit-learn、scipy
运行：python hyperparameter_search.py
"""
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import randint, uniform


def main():
    """GridSearch 与 RandomizedSearch 对比。"""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # ---- GridSearchCV：搜索 SVM 的 C 和 gamma ----
    # Pipeline 中参数用 "步骤名__参数名" 格式
    svm_pipe = make_pipeline(StandardScaler(), SVC())
    param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
    grid = GridSearchCV(svm_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(Xtr, ytr)
    print("GridSearch best params:", grid.best_params_)
    print("GridSearch test acc:", grid.score(Xte, yte))

    # ---- RandomizedSearchCV：搜索 RandomForest 参数 ----
    # 参数分布用 scipy.stats
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": randint(50, 300),       # 树数量
        "max_depth": randint(2, 8),             # 最大深度
        "min_samples_split": randint(2, 8)      # 节点最小分裂样本数
    }
    rnd = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring="accuracy",
                             random_state=42, n_jobs=-1)
    rnd.fit(Xtr, ytr)
    print("RandomizedSearch best params:", rnd.best_params_)
    print("RandomizedSearch test acc:", rnd.score(Xte, yte))


if __name__ == "__main__":
    main()
