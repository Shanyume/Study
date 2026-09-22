# MachineLearning

常见机器学习算法的学习与实现汇总。每个模块尽量保持"算法实现 + 可运行示例 + 数据集"的结构，便于学习和重构。

## 环境准备

```bash
pip install -r requirements.txt
```

> 说明：`DeepLearning_Tutorials` 下部分原始教程基于旧版 Theano / Python 2，当前仅做语法兼容和目录整理，不在默认环境中强制运行。

## 目录索引

| 模块 | 代码位置 | 数据集 / 输入 | 说明 |
|---|---|---|---|
| Iris 多模型对比 | [Iris/iris_classification.py](Iris/iris_classification.py) | sklearn Iris | KNN、逻辑回归、SVM、决策树、随机森林；输出准确率、混淆矩阵与可视化图 |
| Decision Tree | [DecisionTree/id3_c45.py](DecisionTree/id3_c45.py) | 手工小数据集 | NumPy 实现 ID3 / C4.5，支持树可视化 |
| KMeans | [KMeans/kmeans.py](KMeans/kmeans.py) | [KMeans/data.pkl](KMeans/data.pkl) | 实现 KMeans 与二分 KMeans，示例含聚类可视化 |
| KNN | [KNN/UsePythonAndNumPy/kNN.py](KNN/UsePythonAndNumPy/kNN.py) | 手写数字文本样本 | 基于 NumPy 的 KNN 与手写数字测试 |
| Logistic Regression | [LogisticRegression/UsePythonAndNumPy/logistic_regression.py](LogisticRegression/UsePythonAndNumPy/logistic_regression.py) | 二分类手写数字 | 梯度上升实现二分类逻辑回归 |
| Linear Regression | [LinearRegression/linear_regression.py](LinearRegression/linear_regression.py) | 合成数据 | 支持正规方程与梯度下降 |
| Naive Bayes | [NaiveBayes/NaiveBayes.py](NaiveBayes/NaiveBayes.py) | 离散示例数据 | 实现多项式朴素贝叶斯与高斯朴素贝叶斯 |
| PCA | [PCA/pca.py](PCA/pca.py) | NumPy 数组 | 按累计方差贡献率选择主成分 |
| SVM - SMO | [SVM/SVM_by_SMO](SVM/SVM_by_SMO) | Iris 二分类文本数据 | NumPy 实现 SMO SVM |
| SVM - QP | [SVM/SVM_by_QP](SVM/SVM_by_QP) | 合成二维数据 | cvxopt 求解二次规划 SVM |
| Kernel Ridge | [Ridge/kernel_ridge/kernel_ridge.py](Ridge/kernel_ridge/kernel_ridge.py) | Iris 二分类文本数据 | 核岭回归实现 |
| Manifold Learning | [ManifoldLearning/DimensionalityReduction_DataVisualizing/data_visualizing.py](ManifoldLearning/DimensionalityReduction_DataVisualizing/data_visualizing.py) | sklearn Digits | PCA、LDA、Isomap、LLE、t-SNE 等降维可视化 |
| Deep Learning Tutorials | [DeepLearning_Tutorials](DeepLearning_Tutorials) | MNIST / Olivetti Faces | 旧版 Theano/Keras 教程存档，运行前需单独准备环境 |

## 快速运行

```bash
# Iris 多模型对比
python Iris/iris_classification.py

# KMeans
python KMeans/test.py

# KNN 手写数字
python KNN/UsePythonAndNumPy/kNN.py

# SVM SMO
python SVM/SVM_by_SMO/testSVM-SMO.py

# 朴素贝叶斯
python NaiveBayes/test.py
```

## 统一验证脚本

```bash
python examples/run_all.py
```

当前验证包含核心模块的快速冒烟测试，用于确认 Python 3 / NumPy 2 / scikit-learn 1.9 环境下可以运行。

## 贡献

原始仓库与部分代码来自公开学习项目，感谢原作者：

- [wepon](https://github.com/wepe)
- [Gogary](https://github.com/enjoyhot)
- [Locky](https://github.com/junlulocky)
