# 机器学习速查表

## 监督学习

| 算法 | 类型 | 关键参数 | 代码位置 |
|---|---|---|---|
| 线性回归 | 回归 | lr, n_iters | LinearRegression |
| 逻辑回归 | 分类 | alpha, maxCycles | LogisticRegression |
| KNN | 分类 | k | KNN |
| 决策树 | 分类 | max_depth, criterion | DecisionTree / CART |
| SVM | 分类 | C, kernel, gamma | SVM_by_QP / SVM_by_SMO |
| 随机森林 | 分类 | n_estimators, max_depth | RandomForest |
| AdaBoost | 分类 | n_estimators, learning_rate | AdaBoost |
| Gradient Boosting | 分类 | n_estimators, learning_rate | Boosting/GradientBoosting |
| XGBoost | 分类 | n_estimators, max_depth, lr | XGBoost |
| LightGBM | 分类 | n_estimators, max_depth, lr | LightGBM |
| Softmax 回归 | 多分类 | lr, n_iters, l2 | SoftmaxRegression |

## 无监督学习

| 算法 | 类型 | 关键参数 | 代码位置 |
|---|---|---|---|
| KMeans | 硬聚类 | n_clusters, max_iter | KMeans |
| GMM | 软聚类 | n_components, max_iter | GMM |
| DBSCAN | 密度聚类 | eps, min_samples | DBSCAN |
| 层次聚类 | 聚类 | n_clusters, linkage | HierarchicalClustering |
| Mean Shift | 密度聚类 | bandwidth | MeanShift |
| 谱聚类 | 图聚类 | n_clusters, affinity | SpectralClustering |

## 降维

| 算法 | 监督 | 关键参数 | 代码位置 |
|---|---|---|---|
| PCA | 无 | percent | PCA |
| LDA | 有 | n_components | LDA |
| t-SNE | 无 | n_components, perplexity | ManifoldLearning |
| Isomap | 无 | n_neighbors | ManifoldLearning |

## 异常检测

| 算法 | 关键参数 | 代码位置 |
|---|---|---|
| Isolation Forest | contamination | IsolationForest |
| One-Class SVM | nu, kernel | OneClassSVM |

## 深度学习

| 模型 | 用途 | 代码位置 |
|---|---|---|
| MLP | 全分类/回归 | MLP |
| CNN | 图像分类 | CNN_PyTorch |
| RNN | 时序 | RNN |
| LSTM | 时序（长依赖） | LSTM |
| GRU | 时序（简化 LSTM） | GRU |
| Autoencoder | 降维/重构 | Autoencoder |

## 常用公式

### 线性回归
- 正规方程：`w = (X^T X)^(-1) X^T y`
- 梯度下降：`w = w - lr * (1/m) X^T (Xw - y)`

### 逻辑回归
- sigmoid：`σ(z) = 1 / (1 + exp(-z))`
- 损失：交叉熵 `L = -(1/m) Σ [y log(σ(z)) + (1-y) log(1-σ(z))]`

### 决策树
- 信息熵：`H(y) = -Σ p_k log2(p_k)`
- 信息增益：`IG = H(parent) - Σ (n_i/n) H(child_i)`
- 基尼系数：`Gini = 1 - Σ p_k²`

### SVM
- 对偶问题：`max Σ alpha_i - (1/2) Σ Σ alpha_i alpha_j y_i y_j K(x_i,x_j)`
- 约束：`Σ alpha_i y_i = 0, 0 <= alpha_i <= C`

### KMeans
- 目标：`SSE = Σ_k Σ_{x∈C_k} ||x - μ_k||²`
- 质心更新：`μ_k = mean(x ∈ C_k)`

### GMM
- EM E-step：`resp[n,k] = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)`
- EM M-step：`μ_k = (1/N_k) Σ_n resp[n,k] x_n`

## 核函数

| 核 | 公式 |
|---|---|
| linear | `x·y` |
| poly | `(1 + x·y)^p` |
| RBF/Gaussian | `exp(-||x-y||² / (2σ²))` |

## 评估指标

| 指标 | 公式 / 说明 |
|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2 * P * R / (P + R)` |
| ROC-AUC | TPR vs FPR 曲线下面积 |
| MSE | `(1/m) Σ (y - ŷ)²` |
| R² | `1 - SS_res / SS_tot` |
