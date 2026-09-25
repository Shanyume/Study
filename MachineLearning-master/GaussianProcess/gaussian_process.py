#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高斯过程回归
==============

高斯过程是贝叶斯非参数回归方法，输出预测均值和不确定度区间。
适合小数据集上的函数拟合。

核函数：RBF + WhiteKernel（自动学习长度尺度和噪声）
数据集：一维合成正弦函数 + 高斯噪声

依赖：scikit-learn、numpy、matplotlib
运行：python gaussian_process.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pathlib import Path

# 图片保存目录
BASE = Path(__file__).resolve().parent


def main():
    """拟合一维正弦函数并绘制预测区间。"""
    # 生成带噪声的正弦函数
    rng = np.random.RandomState(42)
    X = np.linspace(0, 10, 40).reshape(-1, 1)  # 40 个观测点
    y = np.sin(X).ravel() + rng.normal(0, 0.1, len(X))  # 加高斯噪声

    # 核函数：RBF（平滑）+ WhiteKernel（噪声）
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    # 高斯过程回归器：normalize_y 让目标值标准化，有助于优化
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(X, y)

    # 在更密的网格上预测
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_plot, return_std=True)

    # 绘图：观测点、预测均值、95% 置信区间
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, s=18, label="observed")
    plt.plot(X_plot, y_mean, label="GP mean")
    plt.fill_between(X_plot.ravel(), y_mean-2*y_std, y_mean+2*y_std, alpha=0.2, label="95% CI")
    plt.legend()
    plt.title("Gaussian Process Regression")
    plt.tight_layout()
    plt.savefig(BASE / "gp_regression.png", dpi=120)
    print("kernel:", gp.kernel_)  # 优化后学习到的核参数
    print("figure saved:", BASE / "gp_regression.png")


if __name__ == "__main__":
    main()
