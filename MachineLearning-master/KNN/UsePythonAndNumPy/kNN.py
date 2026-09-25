# '''
# @author: wepon
# @github: https://github.com/wepe
# @blog:   http://blog.csdn.net/u012162613
# '''
# #!/usr/bin/python
# #-*-coding:utf-8-*-
# from numpy import *
# import operator
# from os import listdir

# def classify0(inX, dataSet, labels, k):
#     dataSetSize = dataSet.shape[0]                  
#     diffMat = tile(inX, (dataSetSize,1)) - dataSet      
#     sqDiffMat = diffMat**2
#     sqDistances = sqDiffMat.sum(axis=1)                  
#     distances = sqDistances**0.5
#     sortedDistIndicies = distances.argsort()            
#     classCount={}                                      
#     for i in range(k):
#         voteIlabel = labels[sortedDistIndicies[i]]
#         classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
#     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
#     return sortedClassCount[0][0]

# def img2vector(filename):
#     returnVect = zeros((1,1024))
#     fr = open(filename)
#     for i in range(32):
#         lineStr = fr.readline()
#         for j in range(32):
#             returnVect[0,32*i+j] = int(lineStr[j])
#     return returnVect

# def handwritingClassTest():

#     hwLabels = []
#     trainingFileList = listdir('trainingDigits')          
#     m = len(trainingFileList)
#     trainingMat = zeros((m,1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]                  
#         fileStr = fileNameStr.split('.')[0]                
#         classNumStr = int(fileStr.split('_')[0])          
#         hwLabels.append(classNumStr)
#         trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
     
#     testFileList = listdir('testDigits')       
#     errorCount = 0.0
#     mTest = len(testFileList)
#     for i in range(mTest):
#         fileNameStr = testFileList[i]
#         fileStr = fileNameStr.split('.')[0]     
#         classNumStr = int(fileStr.split('_')[0])
#         vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
#         classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
#         print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
#         if (classifierResult != classNumStr): errorCount += 1.0
#     print("\nthe total number of errors is: %d" % errorCount)
#     print("\nthe total error rate is: %f" % (errorCount/float(mTest)))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kNN（K 近邻）算法的 NumPy 手写实现，并用 32×32 手写数字数据集做识别演示。

算法思想（无显式训练过程，属于惰性学习，全部计算发生在预测时）：
1. 距离度量：计算待分类样本与训练集中每个样本的欧氏距离
       d(x, y) = sqrt( sum_i (x_i - y_i)^2 )
2. 近邻查找：按距离升序排序，取前 k 个最近的邻居
3. 多数表决：这 k 个邻居按各自类别投票，票数最多的类别作为预测结果
   （k 通常取较小的奇数，可避免平票，如 k=3）
"""

import os
import operator
import numpy as np
import sys

def classify0(inX, dataSet, labels, k):
    """
    KNN 分类器
    :param inX: 待分类的样本向量 (1×n)
    :param dataSet: 训练数据集 (m×n)
    :param labels: 训练标签列表 (长度 m)
    :param k: 最近邻个数
    :return: 预测的类别标签
    """
    # 参数校验
    if k <= 0 or k > dataSet.shape[0]:
        raise ValueError("k 必须大于 0 且不超过训练样本数")
    if inX.shape[1] != dataSet.shape[1]:
        raise ValueError("输入样本维度与训练数据不一致")

    dataSetSize = dataSet.shape[0]
    # 训练样本总数 m
    # 计算欧氏距离（使用 broadcasting）
    # 对应公式 d(inX, x_j) = sqrt( sum_i (inX_i - x_ji)^2 )
    # 实现步骤：广播相减得到 (m×n) 差值矩阵 -> 逐元素平方 -> 按行求和 -> 开根号
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedIndices = distances.argsort()   # 按距离升序排列的索引

    # 多数表决：依次取出距离最近的 k 个邻居，统计它们各自类别的票数
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 按投票数降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 取票数最多的类别标签作为预测结果（平票时取排序靠前者）
    return sortedClassCount[0][0]


def img2vector(filename):
    """
    将 32×32 的文本图像转换为 1×1024 的向量
    :param filename: 图像文件路径
    :return: 1×1024 的 numpy 数组
    """
    returnVect = np.zeros((1, 1024))
    # 32×32 的图像按行主序展平为 1024 维特征向量
    try:
        with open(filename, 'r') as fr:
            for i in range(32):
                lineStr = fr.readline()
                if not lineStr:
                    break
                # 确保每行至少 32 个字符（去掉换行符）
                lineStr = lineStr.strip()
                # 每行 32 个字符：'8' 表示黑像素（笔画），'0' 表示白像素（背景）
                # 第 i 行第 j 列的像素放入向量下标 32*i + j（行主序）
                for j in range(min(32, len(lineStr))):
                    returnVect[0, 32 * i + j] = int(lineStr[j])
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，跳过")
    except Exception as e:
        print(f"读取文件 {filename} 时发生错误: {e}")
    return returnVect


def handwritingClassTest(k=3, train_dir='trainingDigits', test_dir='testDigits'):
    """
    手写数字识别测试
    :param k: KNN 中的 k 值
    :param train_dir: 训练数据目录
    :param test_dir: 测试数据目录
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包后的可执行文件
        script_dir = os.path.dirname(sys.executable)
    else:
        # 常规 Python 脚本
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        # 如果仍然获取不到（极少见），则 fallback 到 __file__
        if not script_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))

    if train_dir is None:
        train_dir = os.path.join(script_dir, 'trainingDigits')
    if test_dir is None:
        test_dir = os.path.join(script_dir, 'testDigits')
    # ---------- 加载训练集 ----------
    # 训练文件命名约定："类别数字_样本序号.txt"，如 "0_1.txt" 表示数字 0 的第 1 个样本
    hwLabels = []
    try:
        trainingFileList = [f for f in os.listdir(train_dir)
                            if os.path.isfile(os.path.join(train_dir, f)) and f.endswith('.txt')]
    except FileNotFoundError:
        print(f"错误：训练目录 {train_dir} 不存在")
        return

    m = len(trainingFileList)
    if m == 0:
        print("训练目录中没有找到 .txt 文件")
        return

    trainingMat = np.zeros((m, 1024))
    for i, fileNameStr in enumerate(trainingFileList):
        # 解析文件名，如 "0_1.txt" -> 类别 0
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        filePath = os.path.join(train_dir, fileNameStr)
        trainingMat[i, :] = img2vector(filePath)

    # ---------- 加载测试集并预测 ----------
    try:
        testFileList = [f for f in os.listdir(test_dir)
                        if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.txt')]
    except FileNotFoundError:
        print(f"错误：测试目录 {test_dir} 不存在")
        return

    mTest = len(testFileList)
    if mTest == 0:
        print("测试目录中没有找到 .txt 文件")
        return

    # 逐个测试样本：图像向量化 -> kNN 预测 -> 与真实类别比对并累计错误数
    errorCount = 0.0
    for fileNameStr in testFileList:
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        filePath = os.path.join(test_dir, fileNameStr)
        vectorUnderTest = img2vector(filePath)

        # 如果向量全为0（可能读取失败），直接跳过
        if np.all(vectorUnderTest == 0):
            print(f"警告：文件 {fileNameStr} 读取异常，跳过")
            continue

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print(f"预测结果: {classifierResult}, 真实类别: {classNumStr}")
        if classifierResult != classNumStr:
            errorCount += 1.0

    # 错误率 = 错误样本数 / 测试样本总数，衡量该 kNN 模型在测试集上的错误水平
    print(f"\n总错误数: {int(errorCount)}")
    print(f"错误率: {errorCount / float(mTest):.4f} ({errorCount}/{mTest})")


if __name__ == '__main__':
    # 只调用一次，传入绝对路径
    # 此处 k=3，即取 3 个最近邻做多数表决
    base_dir = os.path.dirname(os.path.abspath(__file__))
    handwritingClassTest(k=3,
                         train_dir=os.path.join(base_dir, 'trainingDigits'),
                         test_dir=os.path.join(base_dir, 'testDigits'))
    