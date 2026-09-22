# 导入修改后的朴素贝叶斯类（假设文件名为 NaiveBayes.py 并位于同一目录）
from NaiveBayes import MultinomialNB, GaussianNB
import numpy as np

if __name__ == "__main__":
    # ---------- 构造训练数据 ----------
    # 原始数据以两个列表分别表示两个特征，每个特征有15个样本
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],  # 特征1取值
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]   # 特征2取值
    ])
    # 转置使每行对应一个样本，形状变为 (15, 2)
    X = X.T
    # 类别标签，对应15个样本
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    # ---------- 1. 训练多项式朴素贝叶斯 ----------
    # 实例化，alpha=1.0 表示 Laplace 平滑，fit_prior=True 从数据估计先验
    nb = MultinomialNB(alpha=1.0, fit_prior=True)
    nb.fit(X, y)  # 训练模型

    # 打印模型参数（注意属性名已改为带下划线的形式）
    print("多项式朴素贝叶斯模型参数：")
    print("alpha =", nb.alpha)                       # 平滑参数
    print("class_prior_ =", nb.class_prior_)        # 每个类别的先验概率（估计值）
    print("classes_ =", nb.classes_)                # 类别标签数组
    # conditional_prob_ 是一个字典：类别 -> 特征索引 -> {特征值: 概率}
    print("conditional_prob_ =", nb.conditional_prob_)

    # 预测单个样本 [2, 4] 的类别
    pred = nb.predict(np.array([2, 4]))
    print("预测样本 [2, 4] 的类别为：", pred)
    print()

    # ---------- 2. 训练高斯朴素贝叶斯 ----------
    # alpha 参数对高斯模型无影响（仅用于与父类接口兼容），此处设为0
    nb1 = GaussianNB(alpha=0.0)
    # 训练并在训练集上预测，打印所有样本的预测结果
    predictions = nb1.fit(X, y).predict(X)
    print("高斯朴素贝叶斯在训练集上的预测结果：")
    print(predictions)