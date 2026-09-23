#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time


class LinearRegression:
    """
    线性回归模型
    支持两种求解方式：
    1. 正规方程（Normal Equation）—— 直接求解析解
    2. 梯度下降（Gradient Descent）—— 迭代优化
    """

    def __init__(self):
        self.weights = None  # 权重向量 w（含偏置）
        self.bias = None     # 偏置 b（当使用梯度下降时单独存储）

    def fit(self, X, y, method='normal_eq', lr=0.01, n_iters=1000):
        """
        训练线性回归模型
        :param X: 训练特征矩阵，形状 (m, n)，m 为样本数，n 为特征数
        :param y: 目标值向量，形状 (m,)
        :param method: 求解方法，'normal_eq' 或 'gradient_descent'
        :param lr: 学习率（仅梯度下降使用）
        :param n_iters: 迭代次数（仅梯度下降使用）
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("特征矩阵的样本数与目标向量长度不一致")

        if method == 'normal_eq':
            self._fit_normal_eq(X, y)
        elif method == 'gradient_descent':
            self._fit_gradient_descent(X, y, lr, n_iters)
        else:
            raise ValueError(f"不支持的求解方法: {method}")

    def _fit_normal_eq(self, X, y):
        """
        正规方程求解：w = (X^T X)^(-1) X^T y
        在 X 前面拼接一列 1，将偏置合并到权重向量中
        """
        m = X.shape[0]
        # 在特征矩阵 X 左侧拼接一列全 1 向量，用于将偏置 b 合并到权重向量中
        # 拼接后 X_b 的形状为 (m, n+1)，第一列全为 1
        X_b = np.hstack([np.ones((m, 1)), X])  # (m, n+1)

        # 正规方程解析解：w = (X^T X)^(-1) X^T y
        # X_b.T @ X_b 计算 X 的转置乘以 X，得到 (n+1, n+1) 的方阵
        # np.linalg.inv() 对该方阵求逆
        # 再右乘 X_b.T @ y 得到最终的权重向量（含偏置）
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        # 权重向量的第一个元素是偏置 b（对应拼接的那列 1）
        self.bias = self.weights[0]
        # 剩余元素是各特征的权重 w1, w2, ..., wn
        self.weights = self.weights[1:]

    def _fit_gradient_descent(self, X, y, lr, n_iters):
        """
        批量梯度下降求解
        """
        m, n = X.shape
        # 初始化权重向量为零向量，偏置为 0
        self.weights = np.zeros(n)
        self.bias = 0.0

        # 记录每次迭代的代价函数值，用于观察收敛情况
        cost_history = []

        for i in range(n_iters):
            # 计算当前参数下的预测值：y_pred = X @ w + b
            y_pred = X @ self.weights + self.bias
            # 计算预测值与真实值的误差向量
            error = y_pred - y

            # 计算均方误差代价函数：J(w,b) = (1/2m) * Σ(y_pred - y)²
            # 乘以 1/2 是为了后续求导时消去平方项的系数 2，简化计算
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            cost_history.append(cost)

            # 计算代价函数对权重 w 的偏导数：dw = (1/m) * X^T @ error
            # X.T 形状为 (n, m)，error 形状为 (m,)，结果 dw 形状为 (n,)
            dw = (1 / m) * (X.T @ error)
            # 计算代价函数对偏置 b 的偏导数：db = (1/m) * Σ(error)
            db = (1 / m) * np.sum(error)

            # 沿梯度反方向更新参数，lr 为学习率控制步长
            self.weights -= lr * dw
            self.bias -= lr * db

        self.cost_history = cost_history

    def predict(self, X):
        """
        预测
        :param X: 特征矩阵，形状 (m, n)
        :return: 预测值，形状 (m,)
        """
        if self.weights is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        X = np.array(X, dtype=np.float64)
        return X @ self.weights + self.bias

    def score(self, X, y):
        """
        计算 R² 决定系数
        R² = 1 - SS_res / SS_tot
        其中 SS_res 是残差平方和，SS_tot 是总平方和
        R² 越接近 1 表示模型拟合越好
        :param X: 特征矩阵
        :param y: 真实目标值
        :return: R² 值，越接近 1 越好
        """
        y = np.array(y, dtype=np.float64).reshape(-1)
        y_pred = self.predict(X)
        # SS_res（残差平方和）：预测值与真实值之差的平方和，衡量模型未能解释的方差
        ss_res = np.sum((y - y_pred) ** 2)
        # SS_tot（总平方和）：真实值与均值之差的平方和，衡量数据的总方差
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # R² = 1 - SS_res/SS_tot，表示模型解释的方差占总方差的比例
        return 1 - ss_res / ss_tot


def generate_data(m=100, noise=10, seed=42):
    """
    生成模拟数据：y = 3 + 4*x1 + 5*x2 + 噪声
    """
    np.random.seed(seed)
    X = np.random.rand(m, 2) * 100
    y = 3 + 4 * X[:, 0] + 5 * X[:, 1] + np.random.randn(m) * noise
    return X, y


def demo_normal_eq():
    """
    演示：正规方程求解线性回归
    """
    print("=" * 50)
    print("正规方程求解线性回归")
    print("=" * 50)

    X, y = generate_data()
    model = LinearRegression()
    model.fit(X, y, method='normal_eq')

    print(f"权重 w: {model.weights}")
    print(f"偏置 b: {model.bias:.4f}")
    print(f"R² 决定系数: {model.score(X, y):.6f}")

    y_pred = model.predict(X[:5])
    print(f"\n前 5 个样本预测 vs 真实值:")
    for i in range(5):
        print(f"  预测: {y_pred[i]:.2f}, 真实: {y[i]:.2f}")


def demo_gradient_descent():
    """
    演示：梯度下降求解线性回归
    """
    print("\n" + "=" * 50)
    print("梯度下降求解线性回归")
    print("=" * 50)

    X, y = generate_data()
    model = LinearRegression()
    model.fit(X, y, method='gradient_descent', lr=0.0001, n_iters=5000)

    print(f"权重 w: {model.weights}")
    print(f"偏置 b: {model.bias:.4f}")
    print(f"R² 决定系数: {model.score(X, y):.6f}")
    print(f"最终代价: {model.cost_history[-1]:.4f}")


def demo_plot():
    """
    演示：一维线性回归可视化
    """
    np.random.seed(0)
    X_1d = np.linspace(0, 10, 80)
    y_1d = 2.5 * X_1d + 1.0 + np.random.randn(80) * 3

    model = LinearRegression()
    model.fit(X_1d.reshape(-1, 1), y_1d, method='normal_eq')

    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_1d, y_1d, s=20, alpha=0.7, label='Data points')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Linear Regression: y = {model.weights[0]:.2f}x + {model.bias:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_regression_fit.png', dpi=150)
    print("\nPlot saved: linear_regression_fit.png")
    plt.close()


def generate_multivariate_data(m=100, n_features=5, noise=1.0, seed=42):
    """
    生成多变量模拟数据：y = w0 + w1*x1 + ... + wn*xn + 噪声
    :param m: 样本数量
    :param n_features: 特征数量
    :param noise: 噪声标准差
    :param seed: 随机种子，保证结果可复现
    :return: X 特征矩阵, y 目标值, true_weights 真实权重, true_bias 真实偏置
    """
    np.random.seed(seed)
    # 生成标准正态分布的特征矩阵，形状 (m, n_features)
    X = np.random.randn(m, n_features)
    # 随机生成真实权重向量，乘以 10 放大数值范围
    true_weights = np.random.randn(n_features) * 10
    # 随机生成真实偏置，乘以 5 放大数值范围
    true_bias = np.random.randn() * 5
    # 生成目标值：y = X @ w + b + 噪声
    # X @ true_weights 计算每个样本的线性组合，形状 (m,)
    # np.random.randn(m) * noise 添加高斯噪声
    y = X @ true_weights + true_bias + np.random.randn(m) * noise
    return X, y, true_weights, true_bias


def benchmark_multivariate():
    """
    多变量线性回归性能对比测试
    对比正规方程和梯度下降在不同特征数和样本数下的表现
    """
    print("=" * 60)
    print("多变量线性回归性能对比测试")
    print("=" * 60)

    # 测试配置
    feature_sizes = [5, 10, 50, 100]
    sample_sizes = [100, 500, 1000]

    results = {
        'normal_eq': {'time': [], 'r2': []},
        'gradient_descent': {'time': [], 'r2': []}
    }

    print("\n测试不同特征数量（固定样本数=500）:")
    print("-" * 60)
    print(f"{'特征数':<8} {'正规方程时间(s)':<15} {'梯度下降时间(s)':<15} {'正规方程R²':<12} {'梯度下降R²':<12}")
    print("-" * 60)

    for n_features in feature_sizes:
        X, y, true_w, true_b = generate_multivariate_data(m=500, n_features=n_features, noise=1.0)

        # 正规方程
        model_ne = LinearRegression()
        start = time.time()
        model_ne.fit(X, y, method='normal_eq')
        time_ne = time.time() - start
        r2_ne = model_ne.score(X, y)

        # 梯度下降（需要调整学习率以适应特征数）
        model_gd = LinearRegression()
        # 学习率随特征数增加而减小，防止梯度爆炸导致发散
        # 经验公式：lr = base_lr / sqrt(n_features)
        lr = 0.01 / np.sqrt(n_features)  # 根据特征数调整学习率
        start = time.time()
        model_gd.fit(X, y, method='gradient_descent', lr=lr, n_iters=5000)
        time_gd = time.time() - start
        r2_gd = model_gd.score(X, y)

        results['normal_eq']['time'].append(time_ne)
        results['normal_eq']['r2'].append(r2_ne)
        results['gradient_descent']['time'].append(time_gd)
        results['gradient_descent']['r2'].append(r2_gd)

        print(f"{n_features:<8} {time_ne:<15.6f} {time_gd:<15.6f} {r2_ne:<12.6f} {r2_gd:<12.6f}")

    print("\n测试不同样本数量（固定特征数=10）:")
    print("-" * 60)
    print(f"{'样本数':<8} {'正规方程时间(s)':<15} {'梯度下降时间(s)':<15} {'正规方程R²':<12} {'梯度下降R²':<12}")
    print("-" * 60)

    for m in sample_sizes:
        X, y, true_w, true_b = generate_multivariate_data(m=m, n_features=10, noise=1.0)

        # 正规方程
        model_ne = LinearRegression()
        start = time.time()
        model_ne.fit(X, y, method='normal_eq')
        time_ne = time.time() - start
        r2_ne = model_ne.score(X, y)

        # 梯度下降
        model_gd = LinearRegression()
        start = time.time()
        model_gd.fit(X, y, method='gradient_descent', lr=0.01, n_iters=5000)
        time_gd = time.time() - start
        r2_gd = model_gd.score(X, y)

        print(f"{m:<8} {time_ne:<15.6f} {time_gd:<15.6f} {r2_ne:<12.6f} {r2_gd:<12.6f}")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时间对比
    axes[0].plot(feature_sizes, results['normal_eq']['time'], 'b-o', label='Normal Equation', linewidth=2)
    axes[0].plot(feature_sizes, results['gradient_descent']['time'], 'r-s', label='Gradient Descent', linewidth=2)
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Training Time Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R² 对比
    axes[1].plot(feature_sizes, results['normal_eq']['r2'], 'b-o', label='Normal Equation', linewidth=2)
    axes[1].plot(feature_sizes, results['gradient_descent']['r2'], 'r-s', label='Gradient Descent', linewidth=2)
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('linear_regression_benchmark.png', dpi=150)
    print("\nBenchmark plot saved: linear_regression_benchmark.png")
    plt.close()

    # 打印总结
    print("\n" + "=" * 60)
    print("性能对比总结:")
    print("=" * 60)
    print("1. 正规方程：")
    print("   - 优点：直接求解，无需调参，适合中小规模数据")
    print("   - 缺点：需要计算矩阵逆，时间复杂度 O(n³)，特征数过多时慢")
    print("   - 适用场景：特征数 < 10000，样本数适中")
    print("\n2. 梯度下降：")
    print("   - 优点：可扩展性强，适合大规模数据，可处理特征数很多的情况")
    print("   - 缺点：需要调学习率和迭代次数，收敛速度依赖数据尺度")
    print("   - 适用场景：特征数 > 10000，或样本数非常大")
    print("\n建议：")
    print("- 特征数较少时优先使用正规方程（简单快速）")
    print("- 特征数较多时使用梯度下降（避免矩阵求逆的数值不稳定）")
    print("- 实际应用中可考虑使用 SGD、Mini-batch GD 等变体加速收敛")


if __name__ == '__main__':
    demo_normal_eq()
    demo_gradient_descent()
    demo_plot()
    benchmark_multivariate()
