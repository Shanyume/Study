"""GridSearchCV 与 RandomizedSearchCV 超参数搜索示例。"""
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import randint, uniform

def main():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # GridSearchCV: SVM
    svm_pipe = make_pipeline(StandardScaler(), SVC())
    param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}
    grid = GridSearchCV(svm_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    print("GridSearch best params:", grid.best_params_)
    print("GridSearch test acc:", grid.score(Xte, yte))

    # RandomizedSearchCV: RandomForest
    rf = RandomForestClassifier(random_state=42)
    param_dist = {"n_estimators": randint(50, 300), "max_depth": randint(2, 8),
                  "min_samples_split": randint(2, 8)}
    rnd = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring="accuracy",
                             random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print("RandomizedSearch best params:", rnd.best_params_)
    print("RandomizedSearch test acc:", rnd.score(Xte, yte))

if __name__ == "__main__":
    main()
