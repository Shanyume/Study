# -*- coding: utf-8 -*-
"""
鸢尾花（Iris）分类案例
========================

使用经典的鸢尾花数据集，演示一个完整的机器学习工作流：
    1. 数据加载与探查
    2. 数据可视化
    3. 特征探索（线性回归拟合，仅用于观察特征关系）
    4. 数据划分
    5. 多模型训练与评估（KNN、逻辑回归、SVM、决策树、随机森林）
    6. 最优模型混淆矩阵
    7. 决策边界可视化（演示用）
    8. 模型预测示例

依赖：scikit-learn、numpy、matplotlib、pandas
运行：python iris_classification.py
"""

import numpy as np                      # 数值计算和数组操作
import pandas as pd                     # 表格数据处理
import matplotlib.pyplot as plt         # 绘图主接口
from matplotlib.colors import ListedColormap  # 决策边界使用的离散颜色映射
from pathlib import Path                # 面向对象的文件路径处理

from sklearn.datasets import load_iris  # 加载 sklearn 自带的鸢尾花数据集
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # 数据划分、交叉验证和分层交叉验证
from sklearn.preprocessing import StandardScaler  # 把特征标准化为均值 0、方差 1
from sklearn.pipeline import make_pipeline  # 把预处理和模型封装成一条流水线
from sklearn.neighbors import KNeighborsClassifier  # K 近邻分类器
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类器
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix  # 准确率、分类报告、混淆矩阵
)
from sklearn.linear_model import LinearRegression  # 线性回归模型（仅用于特征分析可视化）

# 让图片始终保存到脚本所在目录，不受运行位置影响
BASE_DIR = Path(__file__).resolve().parent

def save_and_show(fig, save_path=None):
    """保存图片；在无界面后端下跳过展示以避免不必要的警告"""
    if save_path:                              # 如果传入了保存路径
        fig.savefig(save_path, dpi=120, bbox_inches='tight')  # 把当前 figure 保存为 PNG
        print(f"[图已保存] {save_path}")       # 打印保存位置，方便确认

    if plt.get_backend().lower() != 'agg':     # Agg 是无界面的图片渲染后端
        plt.show()                             # 只有在有界面的后端时才弹出窗口

    plt.close(fig)                             # 关闭 figure 释放内存


# 全局中文显示设置（如系统无中文字体，会自动回退为英文，不影响运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 尝试中文字体，失败则回退
plt.rcParams['axes.unicode_minus'] = False  # 让负号在图中正常显示


# ============================================================
# 1. 数据加载与探查
# ============================================================
def load_and_explore_data():
    """加载鸢尾花数据集并打印基本信息"""
    iris = load_iris()                                                       # 返回一个 Bunch 对象，包含特征、标签、名称等
    df = pd.DataFrame(                         # 把二维特征数组转成 DataFrame，便于查看和处理
        iris.data,                             # 150 行 4 列的特征矩阵
        columns=iris.feature_names             # 4 个特征列名
    )
    df['target'] = iris.target                 # 添加数字标签列，0/1/2 分别对应三种鸢尾花
    df['species'] = df['target'].map(          # 把数字标签映射成可读的物种名称
        {i: name for i, name in enumerate(iris.target_names)}  # 构造 {0: setosa, 1: versicolor, 2: virginica}
    )

    print("=" * 60)                            # 打印分隔线，让终端输出更清晰
    print("1. 数据集基本信息")                 # 当前步骤标题
    print("=" * 60)                            # 打印分隔线
    print(f"样本总数: {len(df)}")              # 输出数据集有多少行
    print(f"特征数量: {len(iris.feature_names)}")  # 输出有多少列特征
    print(f"特征名称: {iris.feature_names}")   # 输出每个特征列的名称
    print(f"类别名称: {list(iris.target_names)}")  # 输出三种鸢尾花名称
    print("\n各类别样本数量:")                 # 提示下面是类别分布
    print(df['species'].value_counts())        # 统计每个物种的样本数量
    print("\n数据描述统计:")                   # 提示下面是数值统计
    print(df[iris.feature_names].describe().round(2))  # 输出均值、标准差、最值和分位数
    print("\n前 5 行数据:")                    # 提示下面是样例数据
    print(df.head())                           # 查看前 5 行，确认数据结构是否正确

    return iris, df                            # 返回原始数据对象和结构化 DataFrame


# ============================================================
# 2. 数据可视化
# ============================================================
def visualize_data(df, feature_names, target_names, save_path=None):
    """绘制鸢尾花数据的两组可视化图，并按类别叠加线性回归拟合线（仅用于观察特征间的相关趋势，不参与分类建模）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 创建一行两列的子图，axes 是两个坐标轴

    colors = ['r', 'g', 'b']                   # 三种鸢尾花对应的散点颜色
    markers = ['o', 's', '^']                  # 三种鸢尾花对应的散点形状

    # 每个子图对应的 (x特征索引, y特征索引)
    feature_pairs = [(0, 1), (2, 3)]           # 左图画萼片长度/宽度，右图画花瓣长度/宽度

    for ax_idx, (xi, yi) in enumerate(feature_pairs):  # 遍历两组特征组合
        ax = axes[ax_idx]                      # 取出当前子图的坐标轴
        for i, name in enumerate(target_names):  # 遍历三个鸢尾花类别
            subset = df[df['target'] == i]     # 筛选出当前类别的所有样本
            x_vals = subset[feature_names[xi]].values  # 取当前类别的 x 特征值
            y_vals = subset[feature_names[yi]].values  # 取当前类别的 y 特征值

            ax.scatter(
                x_vals, y_vals,               # x、y 特征值
                c=colors[i], marker=markers[i], label=name,  # 指定颜色、形状和图例文字
                edgecolors='k', s=50, alpha=0.7  # 黑色描边、点大小 50、70% 不透明度
            )

            # 按类别拟合线性回归线: y = a*x + b（仅用于观察特征相关性，非分类建模）
            if len(x_vals) >= 2:
                reg = LinearRegression()       # 创建一元线性回归模型
                reg.fit(x_vals.reshape(-1, 1), y_vals)  # sklearn 要求特征是二维数组，所以 reshape
                y_pred = reg.predict(x_vals.reshape(-1, 1))  # 用拟合模型预测原有样本

                
                from sklearn.metrics import r2_score  # 局部导入，避免全局污染
                r2 = r2_score(y_vals, y_pred)  # 计算 R²，衡量这条拟合线的解释能力
                # 在特征范围内绘制拟合直线
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)  # 在特征取值范围内生成 100 个    x 值
                y_line = reg.predict(x_line.reshape(-1, 1))  # 用这些 x 值画出对应的直线 y 值
                ax.plot(
                    x_line, y_line,
                    color=colors[i], linestyle='--', linewidth=2, alpha=0.8,
                    label=f'{name} fit (R²={r2:.2f})'
                )

        ax.set_xlabel(feature_names[xi])       # 设置 x 轴标签
        ax.set_ylabel(feature_names[yi])       # 设置 y 轴标签
        ax.set_title(
            f'{feature_names[xi].split(" (")[0]} vs '
            f'{feature_names[yi].split(" (")[0]}'
        )
        ax.legend(fontsize=8)                  # 显示类别和拟合线图例
        ax.grid(alpha=0.3)                     # 加浅色网格，方便读数

    plt.tight_layout()                         # 自动调整子图间距
    save_and_show(fig, save_path)              # 保存或展示图件


def plot_confusion_matrix(cm, target_names, title, save_path=None):
    """绘制混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(6, 5))     # 创建混淆矩阵图窗和坐标轴
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 用蓝色渐变显示矩阵单元格
    ax.figure.colorbar(im, ax=ax)              # 在右侧添加颜色刻度条

    ax.set(
        xticks=np.arange(cm.shape[1]),         # 横轴刻度对应每个预测类别
        yticks=np.arange(cm.shape[0]),         # 纵轴刻度对应每个真实类别
        xticklabels=target_names,              # 横轴显示物种名
        yticklabels=target_names,              # 纵轴显示物种名
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  # 旋转 x 轴标签避免重叠
             rotation_mode="anchor")         # 以标签右端为锚点旋转

    # 在格子里写上数字
    thresh = cm.max() / 2.0                    # 颜色深浅的分界值，决定文字用黑还是白
    for i in range(cm.shape[0]):               # 遍历矩阵每一行，对应真实类别
        for j in range(cm.shape[1]):           # 遍历矩阵每一列，对应预测类别
            ax.text(j, i, format(cm[i, j], 'd'),  # 在第 i 行第 j 列写入样本数量
                    ha="center", va="center",  # 数字居中显示
                    color="white" if cm[i, j] > thresh else "black")  # 深色格子用白字，浅色格子用黑字

    fig.tight_layout()                         # 自动调整布局
    save_and_show(fig, save_path)              # 保存或展示图件


def plot_decision_boundary(model, X, y, title, save_path=None):
    """绘制二维特征空间上的决策边界（仅取前两个特征用于可视化）"""
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # 背景区域的三种浅色
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])   # 样本点的三种深色

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 扩展 x 轴范围，让边界区域更完整
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 扩展 y 轴范围
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # 对每个网格点预测类别
    Z = Z.reshape(xx.shape)                    # 把预测结果恢复成网格形状，方便填色

    fig, ax = plt.subplots(figsize=(7, 5))     # 创建决策边界图窗
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')  # 用浅色填充每个网格点的预测类别
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolor='k', s=40, alpha=0.8) # 黑色描边、点大小 40、80% 不透明度
    ax.set_xlim(xx.min(), xx.max())            # 固定 x 轴显示范围
    ax.set_ylim(yy.min(), yy.max())            # 固定 y 轴显示范围
    ax.set_title(title)                        # 设置图标题
    ax.set_xlabel('Feature 1 (standardized)')  # 说明横轴是标准化后的第一个特征
    ax.set_ylabel('Feature 2 (standardized)')  # 说明纵轴是标准化后的第二个特征

    save_and_show(fig, save_path)


# ============================================================
# 3. 线性回归拟合分析（仅用于探索特征关系，非分类任务）
# ============================================================
def fit_regression_lines(df, feature_names, target_names,
                         x_idx=2, y_idx=3, save_path=None):
    """
    对指定的一对特征做线性回归拟合分析（仅用于探索特征间关系，非分类任务）：
        - 全体样本的整体拟合
        - 每个类别各自的拟合
    并打印回归方程 y = a*x + b 与 R²，绘制拟合直线图。

    参数:
        x_idx, y_idx: 用于回归的特征索引（默认花瓣长度 vs 花瓣宽度）
    """
    x_name = feature_names[x_idx]              # 拿到作为自变量的特征名
    y_name = feature_names[y_idx]              # 拿到作为因变量的特征名
    colors = ['r', 'g', 'b']                   # 三个类别分别使用的颜色

    print("\n" + "=" * 60)
    print(f"线性回归拟合: {y_name} ~ {x_name}")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 6))     # 创建回归拟合图窗

    # ---- 全体样本的整体拟合 ----
    X_all = df[x_name].values.reshape(-1, 1)   # 取全体样本的自变量，并转成 sklearn 需要的二维形状
    y_all = df[y_name].values                  # 取全体样本的因变量
    reg_all = LinearRegression().fit(X_all, y_all)  # 对全体样本拟合一条直线
    y_pred_all = reg_all.predict(X_all)        # 用拟合直线预测所有样本
    from sklearn.metrics import r2_score       # 局部导入 r2_score
    r2_all = r2_score(y_all, y_pred_all)       # 计算整体拟合的 R²
    a_all, b_all = reg_all.coef_[0], reg_all.intercept_  # 取出斜率和截距

    print(
        f"[全体样本] y = {a_all:.4f} * x + {b_all:.4f}  "
        f"(R² = {r2_all:.4f})"
    )

    # ---- 每个类别各自的拟合 ----
    for i, name in enumerate(target_names):    # 分别对三个类别拟合
        subset = df[df['target'] == i]         # 筛选当前类别样本
        x_vals = subset[x_name].values.reshape(-1, 1)  # 当前类别的自变量
        y_vals = subset[y_name].values         # 当前类别的因变量

        reg = LinearRegression().fit(x_vals, y_vals)  # 拟合当前类别自己的直线
        y_pred = reg.predict(x_vals)           # 用这条直线预测当前类别
        r2 = r2_score(y_vals, y_pred)          # 计算当前类别拟合的 R²
        a, b = reg.coef_[0], reg.intercept_    # 取出当前类别的斜率和截距

        print(
            f"[{name:>10s}] y = {a:.4f} * x + {b:.4f}  "
            f"(R² = {r2:.4f})"
        )

        # 散点
        ax.scatter(
            x_vals.ravel(), y_vals,            # 把二维特征数组压回一维来画散点
            c=colors[i], label=name,           # 用颜色区分类别，并写入图例
            edgecolors='k', s=50, alpha=0.7    # 黑色描边、点大小 50、70% 不透明度
        )
        # 拟合直线
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)  # 在当前类别特征范围内生成 100 个 x 值
        y_line = reg.predict(x_line.reshape(-1, 1))  # 预测这些 x 值对应的直线 y 值
        ax.plot(
            x_line, y_line,
            color=colors[i], linestyle='--', linewidth=2,
            label=f'{name} fit (R²={r2:.3f})'
        )

    # 整体拟合直线
    x_line_all = np.linspace(X_all.min(), X_all.max(), 100)  # 在全体特征范围内生成 x 值
    y_line_all = reg_all.predict(x_line_all.reshape(-1, 1))  # 预测整体拟合直线的 y 值
    ax.plot(
        x_line_all, y_line_all,
        color='black', linestyle='-', linewidth=2.5, alpha=0.8,
        label=f'Overall fit (R²={r2_all:.3f})'
    )

    ax.set_xlabel(x_name)                      # 设置 x 轴标签
    ax.set_ylabel(y_name)                      # 设置 y 轴标签
    ax.set_title(f'Linear Regression Fit: {y_name} ~ {x_name}\n(Feature exploration only, not classification)')  # 说明这只是特征探索
    ax.legend(fontsize=20, loc='upper left')    # 在左上角显示类别和拟合线图例
    ax.grid(alpha=0.3)                         # 加浅色网格方便读数

    plt.tight_layout()                         # 自动调整布局
    save_and_show(fig, save_path)              # 保存或展示图件


# ============================================================
# 5. 模型训练与评估
# ============================================================
def build_models():
    """构建待比较的分类模型；基于距离/梯度的模型用 Pipeline 封装标准化以避免交叉验证泄漏；树模型不需要标准化"""
    # ---- K 近邻分类器（KNN） ----
    # 含义: KNN 是基于距离的惰性学习模型。它没有独立的复杂训练阶段，而是保存训练样本，
    #       预测时假设标准化特征空间中距离相近的样本通常属于同一类别。
    # 训练/预测: fit() 保存训练集；predict() 计算新样本到所有训练样本的距离，找最近的 5 个邻居，
    #           并用邻居中出现次数最多的类别作为预测结果。
    # 参数作用: n_neighbors=5 表示参考 5 个最近样本。k 太小容易受异常点影响，太大可能引入较远样本。
    # 配置原因: StandardScaler 把每个特征变成均值 0、方差 1，避免某些特征主导欧氏距离。
    return {
        'KNN (k=5)': make_pipeline(            # 先标准化，再做 K 近邻分类
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)  # 根据最近的 5 个邻居投票
        ),

        # ---- 逻辑回归分类器（Logistic Regression） ----
        # 含义: 逻辑回归是线性分类模型。虽然名字里有“回归”，但它输出类别概率，
        #       对 Iris 的三分类问题会为每个类别学习一组线性权重。
        # 训练/预测: fit() 通过最大似然优化寻找权重和偏置；predict() 选择概率最高的类别。
        # 参数作用: max_iter=200 是优化算法的最大迭代次数，保证较小数据集上有足够机会收敛。
        # 配置原因: StandardScaler 让优化过程更稳定；这里保留默认正则化强度 C=1.0。
        'Logistic Regression': make_pipeline(  # 先标准化，再做逻辑回归分类
            StandardScaler(), LogisticRegression(max_iter=200)  # max_iter 调大，保证收敛
        ),

        # ---- 支持向量机分类器（SVM） ----
        # 含义: 线性 SVM 寻找类别间隔尽量大的决策边界；RBF 核把数据映射到非线性空间，
        #       因此可以学习弯曲的决策边界。
        # 训练/预测: fit() 寻找支持向量和间隔边界；predict() 根据样本相对决策边界的位置分类。
        # 参数作用: C 越大越倾向拟合训练样本，过拟合风险更高；C 越小边界越平滑，欠拟合风险更高。
        #           gamma 控制 RBF 核影响范围，gamma 越大边界越弯曲，gamma 越小边界越平滑。
        # 配置原因: C=1.0 是较保守的默认值；gamma='scale' 按特征方差自动计算，SVM 对尺度敏感，
        #           因此仍在 Pipeline 前段使用 StandardScaler。
        'SVM (RBF)': make_pipeline(            # 先标准化，再训练 RBF 核支持向量机
            StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale')  # C 控制错误容忍度，gamma 自动按特征尺度设置
        ),

        # ---- 决策树分类器（Decision Tree） ----
        # 含义: 决策树用“特征是否小于某个阈值”的规则递归划分样本空间，形成可解释的 if-else 树。
        # 训练/预测: fit() 在每个节点选择更纯的划分方式；predict() 从根节点沿规则走到叶节点。
        # 参数作用: max_depth=4 限制树的最大深度，防止模型记住训练集噪声；random_state=42 保证可复现。
        # 配置原因: 树模型按阈值比较特征，不依赖欧氏距离，同一个特征的阈值分裂不受标准化影响，
        #           因此不额外使用 StandardScaler。
        'Decision Tree': DecisionTreeClassifier(  # 决策树基于特征分裂，不受量纲影响，无需标准化
            max_depth=4, random_state=42        # 限制深度防止过拟合，random_state 保证可复现
        ),

        # ---- 随机森林分类器（Random Forest） ----
        # 含义: 随机森林是决策树的 Bagging 集成模型。每棵树看到随机抽样的数据和随机特征子集，
        #       预测时让多棵树投票，从而降低单棵决策树的不稳定性。
        # 训练/预测: fit() 训练 100 棵有差异的树；predict() 汇总所有树的类别投票。
        # 参数作用: n_estimators=100 表示树的数量；max_depth=4 控制每棵树复杂度；
        #           random_state=42 固定抽样和特征选择过程，保证结果可复现。
        # 配置原因: 随机森林和决策树一样基于阈值分裂，对特征量纲不敏感，因此不做标准化。
        'Random Forest': RandomForestClassifier(  # 随机森林同样基于树分裂，无需标准化
            n_estimators=100, max_depth=4, random_state=42  # 100 棵树，最大深度 4，固定随机种子
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, target_names):
    """训练多个模型并对比其性能"""
    print("\n" + "=" * 60)
    print("5. 多模型训练与评估")
    print("=" * 60)

    models = build_models()                    # 得到所有待比较的分类模型
    results = []                               # 存放每个模型的评估结果

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 分 5 折；每折保持类别比例；固定种子保证可复现

    for name, model in models.items():         # 遍历每个模型
        # 先做 5 折交叉验证（在训练集上评估模型泛化能力）
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring='accuracy'
        )

        # 再用整个训练集训练最终模型
        model.fit(X_train, y_train)            # 用整个训练集训练最终模型
        y_pred = model.predict(X_test)         # 用训练好的模型预测测试集

        acc = accuracy_score(y_test, y_pred)   # 计算测试集整体准确率

        results.append({                       # 记录当前模型的各项指标
            'Model': name,                     # 模型名称
            'Test Accuracy': acc,              # 测试集准确率
            'CV Mean Accuracy': cv_scores.mean(),  # 5 折交叉验证平均准确率
            'CV Std': cv_scores.std(),         # 交叉验证结果的标准差，反映稳定性
        })

        print(f"\n--- {name} ---")             # 打印当前模型标题
        print(f"测试集准确率: {acc:.4f}")      # 输出保留 4 位小数的测试准确率
        print(f"5折交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")  # 输出交叉验证均值和波动
        print("分类报告:")                     # 提示下面是 precision/recall/F1
        print(classification_report(y_test, y_pred, target_names=target_names))  # 输出逐类别评估指标

    results_df = pd.DataFrame(results).sort_values(  # 把结果列表转成 DataFrame
        'CV Mean Accuracy', ascending=False    # 按交叉验证平均准确率从高到低排序
    ).reset_index(drop=True)                   # 重置行号，让第一名是第 0 行

    print("\n" + "=" * 60)
    print("模型性能对比（按交叉验证平均准确率降序）")
    print("=" * 60) 
    print(results_df.to_string(index=False))   # 以表格形式输出模型对比结果

    return models, results_df                  # 返回已训练模型字典和结果表


# ============================================================
# 主流程
# ============================================================
def main():
    # 1. 加载并探查数据
    iris, df = load_and_explore_data()         # 加载原始数据和结构化表格

    # 2. 可视化原始数据
    print("\n" + "=" * 60)                     # 打印步骤分隔线
    print("2. 数据可视化")                     # 步骤标题
    print("=" * 60)                            # 打印步骤分隔线
    visualize_data(
        df, iris.feature_names, iris.target_names,  # 传入 DataFrame、特征名和类别名
        save_path=BASE_DIR / 'iris_scatter.png'  # 指定散点图保存到脚本所在目录
    )

    # 3. 线性回归拟合分析（花瓣长度 vs 花瓣宽度，仅用于探索特征关系）
    print("\n" + "=" * 60)                     # 打印步骤分隔线
    print("3. 线性回归拟合分析（特征探索）")    # 步骤标题
    print("=" * 60)                            # 打印分隔线
    fit_regression_lines(
        df, iris.feature_names, iris.target_names,  # 传入数据和名称
        x_idx=2, y_idx=3,                      # 用第 3 个特征预测第 4 个特征，即花瓣长度到花瓣宽度
        save_path=BASE_DIR / 'iris_regression_fit.png'  # 指定拟合图保存路径
    )

    # 4. 划分训练集与测试集
    X = iris.data                              # 取 4 列特征作为输入 X
    y = iris.target                            # 取数字类别标签作为输出 y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # 30% 做测试集；固定种子；保持三类样本比例一致
    )

    # 5. 训练并评估多个模型
    models, results_df = train_and_evaluate(   # 训练并比较五个模型
        X_train, X_test, y_train, y_test, iris.target_names  # 传入训练/测试数据和类别名
    )

    # 6. 最优模型混淆矩阵
    best_model_name = results_df.iloc[0]['Model']  # # 结果表第一行就是 CV 最优模型  # 结果表第一行就是交叉验证最优模型
    best_model = models[best_model_name]       # 根据名称取出已训练好的最优模型
    y_pred_best = best_model.predict(X_test)   # 用最优模型预测测试集
    cm = confusion_matrix(y_test, y_pred_best)  # # 生成真实类别和预测类别的混淆矩阵 # 生成真实类别和预测类别的交叉表

    print("\n" + "=" * 60)                     # 打印步骤分隔线
    print(f"6. 最优模型 [{best_model_name}] 混淆矩阵")  # 标明当前最优模型
    print("=" * 60)                            # 打印分隔线
    print(cm)                                  # 输出混淆矩阵数值
    plot_confusion_matrix(
        cm, iris.target_names,                 # 传入混淆矩阵和类别名
        f'Confusion Matrix - {best_model_name}',  # 图标题中带上模型名
        save_path=BASE_DIR / 'iris_confusion_matrix.png'  # 指定混淆矩阵图保存路径
    )

    # 7. 决策边界可视化（演示用，非最优模型的真实边界）
    print("\n" + "=" * 60)
    print("7. 决策边界可视化（演示用，非最优模型的真实边界）")
    print("=" * 60)
    scaler = StandardScaler()                  # 为二维边界图单独创建标准化器
    X_train_scaled = scaler.fit_transform(X_train)  # 只在训练集上学习均值和标准差
    X_2d = X_train_scaled[:, :2]  # # 只取前两个特征用于二维可视化               # 只取前两个特征，因为二维平面只能展示两个维度
    boundary_model = KNeighborsClassifier(n_neighbors=5)  # 单独训练一个用于演示的 KNN
    boundary_model.fit(X_2d, y_train)          # 只用前两个特征训练，这是一个独立的二维演示模型
    plot_decision_boundary(
        boundary_model, X_2d, y_train,         # 传入二维模型、数据和标签
        'Decision Boundary Demo (KNN, 2 features only)\nNot the best model — for visualization purposes only',  # 明确说明这只是演示
        save_path=BASE_DIR / 'iris_decision_boundary.png'  # 指定边界图保存路径
    )

    # 8. 用最优模型对多个新样本进行预测
    print("\n" + "=" * 60)
    print("8. 新样本预测示例")
    print("=" * 60)
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],                  # 典型 setosa
        [6.0, 2.7, 5.1, 1.6],                  # 典型 versicolor
        [7.3, 2.9, 6.3, 1.8],                  # 典型 virginica
    ])
    preds = best_model.predict(new_samples)    # 最优模型是 Pipeline，会自动先标准化再预测
    for sample, pred in zip(new_samples, preds):
        print(f"特征: {sample}  →  预测类别: {iris.target_names[pred]}")

    print("\n鸢尾花分类案例运行完毕。")


if __name__ == '__main__':
    main()
