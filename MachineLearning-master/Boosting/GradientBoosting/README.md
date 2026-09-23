# Gradient Boosting

- 实现：[gradient_boosting.py](gradient_boosting.py)
- 简化版决策树桩 + 梯度提升（logistic loss）
- 数据：`sklearn.datasets.make_moons`
- 运行：`python gradient_boosting.py`

## 说明

逐轮拟合当前模型负梯度（残差），逐步修正错误，是 XGBoost / LightGBM 的基础思想。
