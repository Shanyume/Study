#coding:utf-8
"""
流形学习与降维方法对比实验
============================

本脚本在 sklearn 手写数字数据集（digits）上对比多种降维方法的效果。
降维的目标是将高维数据（64 维 = 8×8 图像）映射到低维空间（2D 或 3D），
同时尽量保留数据的内在结构（如类别可分性、局部邻域关系等）。

对比的降维方法分为三类：

1. 线性方法：
   - Random Projection（随机投影）：用随机矩阵投影，保持距离近似不变（Johnson-Lindenstrauss 引理）
   - PCA（主成分分析）：找到方差最大的正交方向进行投影，保留全局方差结构
   - LDA（线性判别分析）：最大化类间散度与类内散度的比值，保留类别可分性（有监督）

2. 流形学习方法（非线性降维）：
   - Isomap：用测地距离（沿流形的最短路径）代替欧氏距离，再做 MDS
   - LLE（局部线性嵌入）：每个点用邻居线性重构，保持局部线性关系
   - Modified LLE：改进的 LLE，使用多重权重向量
   - HLLE（Hessian LLE）：用 Hessian 矩阵估计局部流形结构
   - LTSA（局部切空间对齐）：用局部切空间近似流形，对齐后得到全局嵌入
   - MDS（多维缩放）：直接保持点对之间的距离矩阵
   - Spectral Embedding（谱嵌入）：用图拉普拉斯矩阵的特征向量做嵌入
   - t-SNE（t-分布随机邻域嵌入）：用 KL 散度最小化高维/低维邻域分布的差异

3. 集成方法：
   - Random Trees（随机树嵌入）：用随机森林将数据映射到叶节点指示向量，再降维可视化

参数说明：
   - n_neighbors=30：流形方法中 K 近邻的 K 值，控制局部邻域大小
   - n_components：降维目标维度（2 或 3）

依赖：numpy、matplotlib、sklearn
运行：python data_visualizing.py
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble, random_projection)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ============================================================
# 1. 加载数据
# ============================================================
# 加载 sklearn 手写数字数据集，仅取前 5 类（0~4），共约 900 个样本
# 每个样本是 8×8 的灰度图像，展平为 64 维向量
digits = datasets.load_digits(n_class=5)
X = digits.data    # 特征矩阵 (n_samples, 64)
y = digits.target  # 标签数组 (n_samples,)，值为 0~4
print(X.shape)

# 将前 400 个样本拼成 20×20 的缩略图网格，用于可视化原始数据
# 每张数字图 8×8，间隔 2 像素排列在 200×200 的大图上
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1       # 起始行坐标（留 1 像素间隔）
    for j in range(n_img_per_row):
        iy = 10 * j + 1   # 起始列坐标
        # 将第 i*n_img_per_row+j 个样本 reshape 回 8×8 并放入网格
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.title('A selection from the 64-dimensional digits dataset')

# 流形方法的邻居数参数：控制局部邻域大小
# 太小会丢失全局结构，太大可能混入不同流形的点
n_neighbors = 30


# ============================================================
# 2. 可视化函数
# ============================================================
def plot_embedding_2d(X, title=None):
    """
    将降维后的 2D 数据可视化：在二维平面上用数字字符标记每个样本点

    先将坐标缩放到 [0, 1] 区间，然后用 ax.text 在每个点的位置写上对应的数字标签，
    颜色按类别区分。这样可以直观看到降维后同类数字是否聚在一起。

    :param X: 降维后的数据 (n_samples, 2)
    :param title: 图标题
    """
    # 坐标归一化到 [0, 1]，使所有方法的结果在同一尺度下可比
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        # 在每个样本的 2D 坐标处写上对应的数字标签
        # 颜色由类别决定（Set1 色图），字体加粗、大小 9
        ax.text(X[i, 0], X[i, 1],str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)

def plot_embedding_3d(X, title=None):
    """
    将降维后的 3D 数据可视化：在三维空间中用数字字符标记每个样本点

    :param X: 降维后的数据 (n_samples, 3)
    :param title: 图标题
    """
    # 坐标归一化到 [0, 1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i in range(X.shape[0]):
        # 在 3D 坐标处写上数字标签
        ax.text(X[i, 0], X[i, 1], X[i,2],str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)


# ============================================================
# 3. 随机投影（Random Projection）
# ============================================================
# 原理：Johnson-Lindenstrauss 引理保证，随机矩阵投影后点对距离近似保持
# 优点：计算极快，不需要优化
# 缺点：不保留类别结构，仅保持距离
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding_2d(X_projected, "Random Projection")

# ============================================================
# 4. PCA（主成分分析）
# ============================================================
# 原理：对协方差矩阵做特征分解，取前 k 个最大特征值对应的特征向量作为投影方向
# 优点：保留全局方差结构，计算快，解析解
# 缺点：线性方法，无法捕捉非线性流形结构
print("Computing PCA projection")
t0 = time()
# TruncatedSVD 是 PCA 的稀疏矩阵版本，这里用于降维到 3 维
X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)
plot_embedding_2d(X_pca[:,0:2],"PCA 2D")
plot_embedding_3d(X_pca,"PCA 3D (time %.2fs)" %(time() - t0))

# ============================================================
# 5. LDA（线性判别分析）
# ============================================================
# 原理：最大化类间散度矩阵 S_B 与类内散度矩阵 S_W 的比值
#       即求解 S_W^{-1} S_B 的前 k 个最大特征值对应的特征向量
# 优点：有监督方法，降维后类别可分性更好
# 缺点：线性方法，最多降维到 C-1 维（C 为类别数）
# 注意：需要让 X 可逆（加微小扰动），否则 S_W 不可逆
print("Computing LDA projection")
X2 = X.copy()
# 在对角线上加微小扰动，使 X 满秩，保证 S_W 可逆
X2.flat[::X.shape[1] + 1] += 0.01
t0 = time()
X_lda = LinearDiscriminantAnalysis(n_components=3).fit_transform(X2, y)
plot_embedding_2d(X_lda[:,0:2],"LDA 2D" )
plot_embedding_3d(X_lda,"LDA 3D (time %.2fs)" %(time() - t0))


# ============================================================
# 6. Isomap（等距映射）
# ============================================================
# 原理：
#   1. 用 K 近邻构建邻域图
#   2. 用 Floyd/Dijkstra 算法计算图上所有点对的最短路径（测地距离）
#   3. 对测地距离矩阵做 MDS（多维缩放），得到低维嵌入
# 优点：能捕捉非线性流形结构，保持全局测地距离
# 缺点：对噪声敏感，计算量大（需要全源最短路径）
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding_2d(X_iso,"Isomap (time %.2fs)" %(time() - t0))


# ============================================================
# 7. LLE（局部线性嵌入）- 标准版
# ============================================================
# 原理：
#   1. 找每个点的 K 个邻居
#   2. 用邻居线性重构该点，求重构权重 W（最小化重构误差）
#   3. 固定 W，在低维空间找嵌入 Y，使 Y 也用同样的 W 重构
# 优点：保持局部线性关系，计算相对快
# 缺点：对 K 值敏感，不能外推到新数据
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
plot_embedding_2d(X_lle,"Locally Linear Embedding (time %.2fs)" %(time() - t0))


# ============================================================
# 8. Modified LLE（改进的局部线性嵌入）
# ============================================================
# 原理：使用多重权重向量（每个邻居一个权重），改进标准 LLE 的正则化问题
# 优点：比标准 LLE 更稳定
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
plot_embedding_2d(X_mlle,"Modified Locally Linear Embedding (time %.2fs)" %(time() - t0))


# ============================================================
# 9. HLLE（Hessian 局部线性嵌入）
# ============================================================
# 原理：用 Hessian 矩阵估计局部流形的二阶结构，比 LLE 更好地保持流形几何
# 优点：理论上更严谨，能处理更复杂的流形
# 缺点：计算量大，对 K 值敏感
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
plot_embedding_2d(X_hlle,"Hessian Locally Linear Embedding (time %.2fs)" %(time() - t0))


# ============================================================
# 10. LTSA（局部切空间对齐）
# ============================================================
# 原理：
#   1. 对每个点的邻域用 PCA 估计局部切空间
#   2. 对齐相邻点的切空间，得到全局嵌入
# 优点：保持局部几何结构，理论优雅
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print(("Done. Reconstruction error: %g" % clf.reconstruction_error_))
plot_embedding_2d(X_ltsa,"Local Tangent Space Alignment (time %.2fs)" %(time() - t0))

# ============================================================
# 11. MDS（多维缩放）
# ============================================================
# 原理：直接保持点对之间的欧氏距离，通过最小化 stress 函数优化低维坐标
#       stress = Σ_{i<j} (d_ij - δ_ij)^2 / d_ij^2
#       其中 d_ij 是低维距离，δ_ij 是高维距离
# 优点：直观，保持全局距离结构
# 缺点：计算量大（需要 O(n^2) 距离矩阵），对初始化敏感
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print(("Done. Stress: %f" % clf.stress_))
plot_embedding_2d(X_mds,"MDS (time %.2fs)" %(time() - t0))

# ============================================================
# 12. Random Trees（随机树嵌入）
# ============================================================
# 原理：
#   1. 用 200 棵随机决策树（max_depth=5）将每个样本映射到叶节点
#   2. 每棵树的叶节点构成一个 one-hot 向量，所有树拼接成高维稀疏向量
#   3. 再用 PCA 降维到 2D 可视化
# 优点：无监督特征提取，能捕捉非线性关系
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X)
# 随机树输出的高维稀疏向量，再用 PCA 降到 2D
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding_2d(X_reduced,"Random Trees (time %.2fs)" %(time() - t0))

# ============================================================
# 13. Spectral Embedding（谱嵌入）
# ============================================================
# 原理：
#   1. 构建邻接图（用 RBF 核计算点对相似度）
#   2. 计算图拉普拉斯矩阵 L = D - W（D 为度矩阵，W 为邻接矩阵）
#   3. 取 L 的前 k 个最小非零特征值对应的特征向量作为嵌入
# 优点：保持局部邻域结构，能处理复杂流形
# 缺点：需要求解特征值问题，计算量 O(n^3)
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X)
plot_embedding_2d(X_se,"Spectral (time %.2fs)" %(time() - t0))

# ============================================================
# 14. t-SNE（t-分布随机邻域嵌入）
# ============================================================
# 原理：
#   1. 在高维空间用高斯核计算点对的相似度（条件概率 p_{j|i}）
#   2. 在低维空间用 Student-t 分布（自由度 1，即柯西分布）计算相似度 q_{j|i}
#   3. 最小化 KL 散度：KL(P||Q) = Σ_{i≠j} p_{j|i} log(p_{j|i}/q_{j|i})
# 优点：保持局部邻域结构，可视化效果最好，能揭示簇结构
# 缺点：计算量大 O(n^2)，不能外推到新数据，结果受随机初始化影响
# 注意：init='pca' 用 PCA 初始化，比随机初始化更快收敛且结果更稳定
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)
plot_embedding_2d(X_tsne[:,0:2],"t-SNE 2D")
plot_embedding_3d(X_tsne,"t-SNE 3D (time %.2fs)" %(time() - t0))

# 显示所有图
plt.show()
