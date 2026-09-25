# 机器学习学习路线

本仓库按"从基础到进阶"排列的学习路径，建议按顺序学习。

## 第一阶段：基础算法（1-2 周）

### 1. 线性回归（LinearRegression）
- 学什么：最简单的回归模型，理解"损失函数 + 梯度下降"
- 核心文件：`LinearRegression/linear_regression.py`
- 重点看：
  - 正规方程 `w = (X^T X)^(-1) X^T y`
  - 梯度下降迭代更新
- 动手改：
  - 改学习率 `lr`，观察收敛速度
  - 改迭代次数 `n_iters`
  - 对比两种方法的最终权重

### 2. 逻辑回归（LogisticRegression）
- 学什么：线性分类，sigmoid 函数
- 核心文件：`LogisticRegression/UsePythonAndNumPy/logistic_regression.py`
- 重点看：
  - sigmoid 函数把线性得分映射到 (0,1)
  - 梯度上升最大化对数似然
- 动手改：
  - 改 `alpha`（学习率）
  - 改 `maxCycles`

### 3. KNN
- 学什么：最简单的惰性学习，无训练阶段
- 核心文件：`KNN/UsePythonAndNumPy/kNN.py`
- 重点看：
  - 欧氏距离计算
  - k 个邻居投票
- 动手改：
  - 改 `k` 值，观察准确率变化
  - k 太小/太大分别有什么问题？

### 4. Naive Bayes
- 学什么：概率模型，贝叶斯定理
- 核心文件：`NaiveBayes/NaiveBayes.py`
- 重点看：
  - 先验概率和条件概率
  - 平滑参数 alpha 的作用

## 第二阶段：进阶监督学习（2-3 周）

### 5. 决策树（DecisionTree）
- 学什么：树结构，递归分裂
- 核心文件：`DecisionTree/id3_c45.py`
- 重点看：
  - 熵和信息增益（ID3）
  - 信息增益比（C4.5）
  - 递归构建树
- 对比：`CART/cart.py` 用基尼系数

### 6. SVM
- 学什么：间隔最大化，核方法
- 核心文件：
  - `SVM/SVM_by_QP/SVCQP.py`（QP 求解）
  - `SVM/SVM_by_SMO/SVCSMO.py`（SMO 算法）
- 重点看：
  - 对偶问题和拉格朗日乘子
  - SMO 每次只优化两个 alpha
  - 核函数把数据映射到高维空间

### 7. 随机森林（RandomForest）
- 学什么：Bagging 集成
- 核心文件：`RandomForest/random_forest.py`
- 重点看：
  - bootstrap 采样
  - 每棵树随机选特征
  - 投票

### 8. Boosting
- 学什么：Boosting 集成，逐轮拟合残差
- 核心文件：
  - `AdaBoost/adaboost.py`
  - `Boosting/GradientBoosting/gradient_boosting.py`
- 重点看：
  - AdaBoost：调整样本权重
  - Gradient Boosting：拟合负梯度
  - 和 Bagging 的区别：串行 vs 并行

## 第三阶段：无监督学习（1-2 周）

### 9. KMeans
- 学什么：硬聚类
- 核心文件：`KMeans/kmeans.py`
- 重点看：
  - 初始化、分配、更新三步
  - SSE（簇内平方误差）

### 10. GMM
- 学什么：软聚类，EM 算法
- 核心文件：`GMM/gmm.py`
- 重点看：
  - E-step：计算响应度
  - M-step：更新参数
  - 和 KMeans 的对比

### 11. DBSCAN
- 学什么：密度聚类，自动发现簇数
- 核心文件：`DBSCAN/dbscan.py`
- 重点看：
  - eps 邻域和核心点
  - 密度可达扩展

### 12. 降维（PCA / LDA）
- 核心文件：
  - `PCA/pca.py`
  - `LDA/lda.py`
- 重点看：
  - PCA：无监督，找方差最大方向
  - LDA：有监督，找类别可分方向

## 第四阶段：深度学习（2-4 周）

### 13. MLP
- 核心文件：`MLP/mlp.py`
- 重点看：前向传播、反向传播

### 14. CNN
- 核心文件：`CNN_PyTorch/cnn_mnist.py`
- 重点看：卷积、池化、全连接

### 15. RNN / LSTM / GRU
- 核心文件：`RNN/`、`LSTM/`、`GRU/`
- 重点看：时序处理，三种循环网络对比

### 16. Autoencoder
- 核心文件：`Autoencoder/autoencoder.py`
- 重点看：编码器-解码器结构

## 第五阶段：模型评估与工程化

### 17. 模型评估
- 核心文件：`Evaluation/model_evaluation.py`
- 重点看：ROC 曲线、PR 曲线、AUC

### 18. 超参数搜索
- 核心文件：`HyperparameterSearch/hyperparameter_search.py`
- 重点看：GridSearchCV vs RandomizedSearchCV

## 学习方法建议

1. **先跑起来**：每个模块先运行示例脚本，看输出
2. **读代码**：打开核心文件，对照模块 README 和 docstring
3. **动手改**：
   - 改参数（学习率、迭代次数、k 值等）
   - 换数据集
   - 加可视化
4. **对比学习**：
   - KMeans vs GMM（硬聚类 vs 软聚类）
   - PCA vs LDA（无监督 vs 有监督降维）
   - Bagging vs Boosting（并行 vs 串行）
   - RNN vs LSTM vs GRU
5. **写笔记**：在 notebooks/ 目录创建自己的学习笔记

## 快速验证

所有模块都可以通过统一脚本验证：

```bash
# 核心模块
python examples/run_all.py

# 新增模块
python examples/run_new_modules.py
```
