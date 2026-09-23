#!/usr/bin/env python3
"""新补充模块的统一冒烟测试。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run(name, script):
    import subprocess
    env = {"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    r = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True, env=env, timeout=180)
    status = "OK" if r.returncode == 0 else "FAIL"
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f"{name}: {status}")
    if r.returncode != 0 and tail:
        print("   ", tail[-1][:150])

if __name__ == "__main__":
    print("=" * 50)
    print("新增模块冒烟测试")
    print("=" * 50)
    run("GradientBoosting", "Boosting/GradientBoosting/gradient_boosting.py")
    run("XGBoost", "XGBoost/xgboost_example.py")
    run("LightGBM", "LightGBM/lightgbm_example.py")
    run("Bagging", "Bagging/bagging.py")
    run("CART", "CART/cart.py")
    run("GMM", "GMM/gmm.py")
    run("HierarchicalClustering", "HierarchicalClustering/hierarchical_clustering.py")
    run("MeanShift", "MeanShift/mean_shift.py")
    run("SpectralClustering", "SpectralClustering/spectral_clustering.py")
    run("IsolationForest", "IsolationForest/isolation_forest.py")
    run("OneClassSVM", "OneClassSVM/one_class_svm.py")
    run("SoftmaxRegression", "SoftmaxRegression/softmax_regression.py")
    run("MulticlassStrategies", "MulticlassStrategies/multiclass_strategies.py")
    run("GaussianProcess", "GaussianProcess/gaussian_process.py")
    run("ModelEvaluation", "Evaluation/model_evaluation.py")
    run("HyperparameterSearch", "HyperparameterSearch/hyperparameter_search.py")
    print("\n完成。")
