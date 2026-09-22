# -*- coding: utf-8 -*-
"""
Created on Oct 14, 2010
@author: Peter Harrington
From the book <<Machine Learning in Action>>
此代码用于绘制决策树（字典结构）的图形，使用 Matplotlib 的 annotate 功能。
"""

import matplotlib.pyplot as plt

# ------------------- 绘图样式配置 -------------------
# 决策节点（内部节点）的文本框样式：锯齿边框，背景色浅灰
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 叶子节点的文本框样式：圆角边框，背景色浅灰
leafNode = dict(boxstyle="round4", fc="0.8")
# 箭头样式：带 '<-' 标记
arrow_args = dict(arrowstyle="<-")


# ------------------- 辅助函数：统计树的叶子节点数和深度 -------------------
def getNumLeafs(myTree):
    """
    递归计算决策树的叶子节点总数。
    参数 myTree：决策树字典，格式如 {'特征名': {分支值: 子节点/类别}}
    返回：叶子节点个数
    """
    numLeafs = 0
    # 获取当前节点的第一个特征名（Python3 中 keys() 返回视图，需转为 list）
    firstStr = list(myTree.keys())[0]
    # 获取该特征对应的分支字典
    secondDict = myTree[firstStr]
    # 遍历每个分支
    for key in secondDict.keys():
        # 如果分支的值仍然是字典，说明是内部节点，继续递归
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   # 否则是叶子节点（类别标签）
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    递归计算决策树的最大深度（从根到叶子的最长路径边数）。
    参数 myTree：决策树字典
    返回：最大深度
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 内部节点：深度加1，继续递归
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   # 叶子节点：深度记为1（当前边）
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# ------------------- 绘图核心函数 -------------------
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    在图上绘制一个节点（带文本框和箭头）。
    参数：
        nodeTxt  : 节点显示的文本
        centerPt : 文本框的中心坐标（子节点位置）
        parentPt : 箭头起点坐标（父节点位置）
        nodeType : 文本框样式（decisionNode 或 leafNode）
    利用全局的 createPlot.ax1 作为绘图区，使用 annotate 添加带箭头的注释。
    """
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,               # 箭头终点（指向父节点）
                            xycoords='axes fraction',  # 坐标以子图相对位置表示
                            xytext=centerPt,           # 文本框位置（子节点处）
                            textcoords='axes fraction',
                            va="center", ha="center",  # 文本垂直、水平居中
                            bbox=nodeType,             # 文本框样式
                            arrowprops=arrow_args)     # 箭头样式


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父节点和子节点连线的中点位置添加文本（通常标注分支条件值）。
    参数：
        cntrPt    : 子节点坐标（当前节点）
        parentPt  : 父节点坐标
        txtString : 要显示的文本（如 '0' 或 '1'）
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString,
                        va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    递归绘制整棵决策树（核心函数）。
    参数：
        myTree   : 当前子树字典
        parentPt : 父节点的坐标
        nodeTxt  : 从父节点到当前节点的分支条件值（用于标注在连线上）
    使用的全局变量（通过 createPlot 设置）：
        plotTree.totalW : 整棵树的叶子节点总数（用于水平方向布局）
        plotTree.totalD : 整棵树的最大深度（用于垂直方向布局）
        plotTree.xOff   : 当前已绘制的叶子节点在水平方向上的偏移（归一化）
        plotTree.yOff   : 当前所在的深度位置（从上到下递减）
    """
    # 计算当前子树的叶子数，用于确定当前节点的水平位置（宽度占比）
    numLeafs = getNumLeafs(myTree)
    # 获取当前节点的特征名（即分裂属性）
    firstStr = list(myTree.keys())[0]
    # 计算当前节点的中心 x 坐标：
    # xOff 是已经绘制的叶子数占总宽度的偏移，加上本子树叶子数的一半偏移，
    # 使得当前节点位于其所有叶子节点的水平中心位置。
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    # 在父节点和当前节点之间添加分支条件文本
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制当前节点（决策节点）
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 获取该节点的分支字典
    secondDict = myTree[firstStr]
    # 进入下一层：y 坐标下移一个深度单位
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    # 遍历每个分支
    for key in secondDict.keys():
        # 如果分支的值仍是字典，则递归绘制子树
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:   # 如果是叶子节点，绘制叶子并标注标签
            # 叶子节点水平方向向右移动一个叶子宽度单位
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 绘制叶子节点（使用叶子样式）
            plotNode(secondDict[key],
                     (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            # 在连线中点添加分支条件文本
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 当前层绘制完毕后，将 y 坐标恢复上层（回溯），以便递归返回时正确处理兄弟节点
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    外部调用接口：创建绘图窗口，设置全局布局变量，并调用 plotTree 绘制。
    参数 inTree：待绘制的决策树字典。
    """
    # 创建画布，背景色白色
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # 设置坐标轴属性：隐藏刻度
    axprops = dict(xticks=[], yticks=[])
    # 创建子图（占满整个画布），frameon=False 去除边框
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 计算整棵树的叶子数和深度，并存储为 plotTree 函数的全局属性
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 初始化水平偏移：从 -0.5/总叶子数 开始，使得绘图时左右留出边距
    plotTree.xOff = -0.5 / plotTree.totalW
    # 初始垂直偏移：从顶部开始（1.0）
    plotTree.yOff = 1.0
    # 开始递归绘制，根节点的父节点位置设为 (0.5, 1.0)（即画布顶部中央）
    plotTree(inTree, (0.5, 1.0), '')
    # 显示图形
    plt.show()


# ------------------- 测试部分 -------------------
if __name__ == '__main__':
    # 构造一个简单的决策树字典（示例）
    # 结构：根节点 'no surfacing?'（是否浮出水面），
    #       分支 0 -> 'no'（叶子），分支 1 -> 内部节点 'flippers?'（是否有脚蹼）
    #       内部节点分支 0 -> 'no'，分支 1 -> 'yes'
    myTree = {'no surfacing?': {0: 'no', 1: {'flippers?': {0: 'no', 1: 'yes'}}}}
    # 调用绘制函数
    createPlot(myTree)