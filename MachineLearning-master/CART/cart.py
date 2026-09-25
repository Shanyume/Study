"""CART 决策树独立实现（基尼系数，二叉分裂）。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CARTClassifier:
    """CART 分类树：使用基尼系数（Gini）作为划分标准，每次分裂成左右两个子节点（二叉树）。

    使用方式：
        - 训练：model = CARTClassifier(max_depth=4).fit(X, y)
        - 预测：pred = model.predict(X)
      其中 X、y 均按样本数组处理，y 为整数类别标签。
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        # max_depth：树的最大深度，用于防止过拟合；min_samples_split：节点继续分裂所需的最少样本数
        self.max_depth = max_depth; self.min_split = min_samples_split

    def _gini(self, y):
        """计算标签数组 y 的基尼不纯度：1 - Σ p_k^2，值越小说明类别越纯。"""
        _, counts = np.unique(y, return_counts=True)  # 统计各类别出现的次数
        p = counts / len(y)                           # 各类别的占比 p_k
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        """遍历所有特征及其取值，寻找使加权基尼不纯度最小的最优划分。

        返回 (特征索引 j, 划分阈值 t, 对应的加权基尼值)；若无可行划分则返回 (None, None, inf)。
        """
        best = (None, None, np.inf)                   # 记录当前最优划分
        for j in range(X.shape[1]):                   # 逐个特征尝试
            for t in np.unique(X[:, j]):              # 以该特征的每个取值作为候选阈值
                left, right = y[X[:, j] <= t], y[X[:, j] > t]  # 按 x_j <= t 划分左右子集的标签
                if len(left) < self.min_split or len(right) < self.min_split: continue  # 任一侧样本过少则跳过
                # 按样本量加权的基尼不纯度
                g = (len(left)*self._gini(left) + len(right)*self._gini(right)) / len(y)
                if g < best[2]: best = (j, t, g)      # 更新最优划分
        return best

    def _build(self, X, y, depth):
        """递归构建决策树，返回表示节点的字典（叶子节点或内部节点）。"""
        # 达到最大深度或节点内类别已唯一时，作为叶子，取多数类作为预测标签
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "label": np.bincount(y).argmax()}  # # 叶节点：取样本数最多的类别
        j, t, g = self._best_split(X, y)              # 寻找当前节点的最优划分
        if j is None: return {"leaf": True, "label": np.bincount(y).argmax()}  # # 叶节点：取样本数最多的类别  # 无可行划分则退化为叶子
        # 递归构建左右子树（依据 x_j <= t 划分样本）
        left = self._build(X[X[:, j] <= t], y[X[:, j] <= t], depth+1)
        right = self._build(X[X[:, j] > t], y[X[:, j] > t], depth+1)
        return {"leaf": False, "j": j, "t": t, "left": left, "right": right}

    def fit(self, X, y):
        """训练入口：由根节点开始构建整棵树，返回 self 以支持链式调用。"""
        self.tree = self._build(np.asarray(X), np.asarray(y), 0); return self

    def _predict_one(self, x, node):
        """对单个样本 x，从给定节点出发沿树下行，返回落入叶子节点的类别标签。"""
        while not node["leaf"]:
            # 根据划分特征与阈值决定走向左子树还是右子树
            node = node["left"] if x[node["j"]] <= node["t"] else node["right"]
        return node["label"]

    def predict(self, X):
        """对样本集 X 逐行预测，返回类别标签数组。"""
        return np.array([self._predict_one(x, self.tree) for x in X])

if __name__ == "__main__":
    # 加载鸢尾花数据集，X 为特征、y 为类别标签
    X, y = load_iris(return_X_y=True)
    # 按 7:3 划分训练集与测试集，stratify=y 保证各类别比例一致
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    # 构建并训练模型（最大深度 4），随后在测试集上评估准确率
    model = CARTClassifier(max_depth=4).fit(Xtr, ytr)
    print("CART test acc:", accuracy_score(yte, model.predict(Xte)))
