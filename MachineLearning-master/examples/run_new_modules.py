#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新增模块统一冒烟测试
======================

对本次新增的所有机器学习模块做快速验证：
- 每个模块作为独立脚本运行
- 输出 OK / FAIL 状态
- FAIL 时打印最后一行错误信息

用法：python examples/run_new_modules.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run(name, script):
    """运行单个脚本并打印状态。"""

    import subprocess
    env = {"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin:/usr/local/bin"}  # # 设置环境变量：无界面后端 + 项目路径
    r = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True, env=env, timeout=180)  # # 运行脚本，捕获输出，超时 180 秒
    status = "OK" if r.returncode == 0 else "FAIL"  # # 根据返回码判断成功或失败
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f"{name}: {status}")
    if r.returncode != 0 and tail:
        print("   ", tail[-1][:150])

if __name__ == "__main__":
    # 依次运行所有新增模块的冒烟测试
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
