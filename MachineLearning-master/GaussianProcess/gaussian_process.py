"""高斯过程回归示例：一维函数拟合 + 不确定度可视化。"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pathlib import Path

BASE = Path(__file__).resolve().parent

def main():
    rng = np.random.RandomState(42)
    X = np.linspace(0, 10, 40).reshape(-1, 1)
    y = np.sin(X).ravel() + rng.normal(0, 0.1, len(X))
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42).fit(X, y)
    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    y_mean, y_std = gp.predict(X_plot, return_std=True)
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, s=18, label="observed")
    plt.plot(X_plot, y_mean, label="GP mean")
    plt.fill_between(X_plot.ravel(), y_mean-2*y_std, y_mean+2*y_std, alpha=0.2, label="95% CI")
    plt.legend(); plt.title("Gaussian Process Regression")
    plt.tight_layout()
    plt.savefig(BASE / "gp_regression.png", dpi=120)
    print("kernel:", gp.kernel_)
    print("figure saved:", BASE / "gp_regression.png")

if __name__ == "__main__":
    main()
