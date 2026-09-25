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
    # 真实函数 f(x) = sin(x)，观测值 = f(x) + 噪声，用于验证 GP 的曲线拟合与不确定度估计
    rng = np.random.RandomState(42)
    X = np.linspace(0, 10, 40).reshape(-1, 1)  # 40 个观测点
    y = np.sin(X).ravel() + rng.normal(0, 0.1, len(X))  # 加高斯噪声

    # 核函数：RBF（平滑）+ WhiteKernel（噪声）
    # RBF 核 k(x, x') = exp(-||x-x'||² / (2*length_scale²))，length_scale 越大曲线越平滑（变化越慢）
    # WhiteKernel 建模独立同分布观测噪声 σ²：k(x,x) 处加白噪声，k(x,x') (x≠x') 处为 0
    # "+" 表示两个核函数相加，分别刻画函数本体的相关性与观测噪声
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    # 高斯过程回归器：normalize_y 让目标值标准化，有助于优化
    # 训练时对核参数做最大边缘似然优化（内部调用 minimize），因此初始值只是起点
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    gp.fit(X, y)

    # 在更密的网格上预测
    # GP 后验为高斯分布：返回均值（点预测）和标准差（预测不确定度）
    # 不确定度在训练点附近小、在远离观测的区域变大——这正是 GP 的核心优势
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_plot, return_std=True)

    # 绘图：观测点、预测均值、95% 置信区间
    # 高斯分布下 2 倍标准差约覆盖 95% 的概率质量
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
