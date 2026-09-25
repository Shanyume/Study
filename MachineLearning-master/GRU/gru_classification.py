#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU（门控循环单元）时序分类
============================

GRU 是 LSTM 的简化变体，用更少的门（更新门 + 重置门）实现类似的记忆能力。
参数更少、训练更快，在很多时序任务上效果接近 LSTM。

本脚本用合成数据演示 GRU 的时序二分类：
    - 类别 0：正弦波
    - 类别 1：余弦波
每条序列长度 50，带随机相位偏移。

依赖：PyTorch、numpy、scikit-learn
运行：python gru_classification.py
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SimpleGRU(nn.Module):
    """单层 GRU + 全连接分类器。"""

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        # GRU 层：input_size 个特征每步输入，hidden_size 个隐状态
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层：把最后的隐状态映射到 2 类
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        """前向传播：GRU 输出后取最后一个时间步的隐状态做分类。"""
        out, _ = self.gru(x)       # out: (B, T, hidden_size)
        return self.fc(out[:, -1, :]) # 取最后时刻，(B, hidden_size) -> (B, 2)


def make_data(n=1000, length=50):
    """
    生成合成时序数据。
    :param n: 样本总数（一半正弦、一半余弦）
    :param length: 每条序列的时间步数
    :return: X 形状 (n, length, 1)，y 形状 (n,)
    """
    # 时间轴：0 到 2π，均匀采样 length 个点
    t = np.linspace(0, 2*np.pi, length)
    # 正弦波样本：每条加随机相位偏移，模拟多样性
    X_sin = np.array([np.sin(t + np.random.rand()*0.5) for _ in range(n//2)])
    # 余弦波样本
    X_cos = np.array([np.cos(t + np.random.rand()*0.5) for _ in range(n//2)])
    # 拼接成 (n, length)，再加一维作为特征维度 -> (n, length, 1)
    X = np.vstack([X_sin, X_cos])[:, :, None].astype(np.float32)
    # 标签：前半是 0（正弦），后半是 1（余弦）
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y


def main():
    """训练 GRU 30 个 epoch，输出测试集准确率。"""
    # 生成数据并划分训练/测试集
    X, y = make_data(1000, 50)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    model = SimpleGRU()
    # 交叉熵损失：适用于二分类
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 转成 PyTorch tensor
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)

    # 训练 30 个 epoch
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()       # 清空梯度
        loss = criterion(model(Xtr_t), ytr_t)  # 前向 + 计算损失
        loss.backward()             # 反向传播
        optimizer.step()            # 更新参数

    # 测试：关闭 dropout 等训练专用行为
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(dim=1).numpy()
    print("GRU test acc:", accuracy_score(yte, pred))


if __name__ == "__main__":
    main()
