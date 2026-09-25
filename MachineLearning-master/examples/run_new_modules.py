#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新增模块统一冒烟测试
======================

对本次新增的所有机器学习模块做快速验证，确保每个模块在当前环境下能正常运行。

测试方式：
    - 每个模块作为独立子进程运行（subprocess.run）
    - 设置环境变量：MPLBACKEND=Agg（无界面后端，避免弹窗）、PYTHONPATH=项目根目录
    - 超时 180 秒自动终止
    - 根据返回码判断成功（OK）或失败（FAIL）
    - FAIL 时打印最后一行输出，方便快速定位错误

测试覆盖的模块（共 16 个）：
    集成学习：GradientBoosting、XGBoost、LightGBM、Bagging
    决策树：CART
    聚类：GMM、HierarchicalClustering、MeanShift、SpectralClustering
    异常检测：IsolationForest、OneClassSVM
    回归/分类：SoftmaxRegression、MulticlassStrategies、GaussianProcess
    工具：ModelEvaluation、HyperparameterSearch

用法：python examples/run_new_modules.py
"""
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使各模块可以被导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run(name, script):
    """
    运行单个模块脚本并打印状态

    :param name: 模块名称（用于打印）
    :param script: 脚本相对路径（相对于项目根目录）
    """
    import subprocess
    # 设置子进程环境变量：
    # - PYTHONPATH: 项目根目录，确保模块可被导入
    # - MPLBACKEND=Agg: 无界面后端，避免 matplotlib 弹窗
    # - PATH: 基础系统路径
    env = {"PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    # 运行脚本，捕获 stdout 和 stderr，超时 180 秒
    r = subprocess.run([sys.executable, str(ROOT / script)], capture_output=True, text=True, env=env, timeout=180)
    # 根据返回码判断成功或失败
    status = "OK" if r.returncode == 0 else "FAIL"
    # 提取输出的最后一行（通常是错误信息或最终结果）
    tail = (r.stdout + r.stderr).strip().splitlines()
    print(f"{name}: {status}")
    # 失败时打印最后一行输出，截断到 150 字符避免过长
    if r.returncode != 0 and tail:
        print("   ", tail[-1][:150])

if __name__ == "__main__":
    # 依次运行所有新增模块的冒烟测试
    print("=" * 50)
    print("新增模块冒烟测试")
    print("=" * 50)
    # ---- 集成学习 ----
    run("GradientBoosting", "Boosting/GradientBoosting/gradient_boosting.py")
    run("XGBoost", "XGBoost/xgboost_example.py")
    run("LightGBM", "LightGBM/lightgbm_example.py")
    run("Bagging", "Bagging/bagging.py")
    # ---- 决策树 ----
    run("CART", "CART/cart.py")
    # ---- 聚类 ----
    run("GMM", "GMM/gmm.py")
    run("HierarchicalClustering", "HierarchicalClustering/hierarchical_clustering.py")
    run("MeanShift", "MeanShift/mean_shift.py")
    run("SpectralClustering", "SpectralClustering/spectral_clustering.py")
    # ---- 异常检测 ----
    run("IsolationForest", "IsolationForest/isolation_forest.py")
    run("OneClassSVM", "OneClassSVM/one_class_svm.py")
    # ---- 回归/分类 ----
    run("SoftmaxRegression", "SoftmaxRegression/softmax_regression.py")
    run("MulticlassStrategies", "MulticlassStrategies/multiclass_strategies.py")
    run("GaussianProcess", "GaussianProcess/gaussian_process.py")
    # ---- 工具 ----
    run("ModelEvaluation", "Evaluation/model_evaluation.py")
    run("HyperparameterSearch", "HyperparameterSearch/hyperparameter_search.py")
    print("\n完成。")
