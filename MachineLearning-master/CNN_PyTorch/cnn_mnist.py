#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN（卷积神经网络）MNIST 手写数字分类
========================================

本脚本用 PyTorch 实现一个简洁的 CNN 模型，用于 MNIST 手写数字识别。

CNN 核心思想：
    - 卷积层（Conv2d）：用可学习的卷积核在输入图像上滑动，提取局部特征（如边缘、纹理）
    - 池化层（MaxPool2d）：对局部区域取最大值，降低空间分辨率，提供平移不变性
    - 全连接层（Linear）：将提取的特征展平后做分类决策

网络结构：
    输入: (B, 1, 28, 28) — 批量大小 × 1 通道 × 28×28 像素
    ↓
    Conv2d(1→32, 3×3, padding=1) → ReLU → MaxPool(2×2)
    输出: (B, 32, 14, 14) — 32 个特征图，空间尺寸减半
    ↓
    Conv2d(32→64, 3×3, padding=1) → ReLU → MaxPool(2×2)
    输出: (B, 64, 7, 7) — 64 个特征图，空间尺寸再减半
    ↓
    Flatten → Linear(3136→128) → ReLU → Linear(128→10)
    输出: (B, 10) — 10 个类别的 logits

训练细节：
    - 损失函数：CrossEntropyLoss（内部包含 softmax + 负对数似然）
    - 优化器：Adam（自适应学习率，lr=1e-3）
    - 批量大小：训练 128，测试 512
    - 训练轮数：3 个 epoch

依赖：PyTorch、torchvision、scikit-learn
运行：python cnn_mnist.py（首次运行会自动下载 MNIST 数据集）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score


class SimpleCNN(nn.Module):
    """
    简洁 CNN：两层卷积 + 两层全连接

    架构说明：
        - 第一个卷积块：32 个 3×3 卷积核 → ReLU → 2×2 最大池化
          作用：提取低级特征（边缘、角点等）
        - 第二个卷积块：64 个 3×3 卷积核 → ReLU → 2×2 最大池化
          作用：在低级特征基础上提取更高级的特征（纹理、形状等）
        - 全连接块：展平 → 128 维隐藏层 → 10 类输出
          作用：将特征图映射到类别空间
    """

    def __init__(self):
        super().__init__()
        # 用 Sequential 将各层串联，前向传播时按顺序执行
        self.net = nn.Sequential(
            # 第一个卷积层：1 通道输入 → 32 通道输出，3×3 卷积核
            # padding=1 保持空间尺寸不变（28×28 → 28×28）
            # 卷积核参数数量：3×3×1×32 + 32(偏置) = 320
            nn.Conv2d(1, 32, 3, padding=1),
            # ReLU 激活函数：引入非线性，f(x) = max(0, x)
            nn.ReLU(),
            # 2×2 最大池化：空间尺寸减半（28×28 → 14×14）
            # 作用：降低计算量，提供平移不变性
            nn.MaxPool2d(2),

            # 第二个卷积层：32 通道输入 → 64 通道输出，3×3 卷积核
            # padding=1 保持尺寸（14×14 → 14×14）
            # 卷积核参数数量：3×3×32×64 + 64 = 18496
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            # 2×2 最大池化：14×14 → 7×7
            nn.MaxPool2d(2),

            # 展平层：将 (B, 64, 7, 7) 展平为 (B, 64×7×7) = (B, 3136)
            nn.Flatten(),
            # 第一个全连接层：3136 → 128，提取高层特征组合
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            # 输出层：128 → 10，对应 10 个数字类别（0~9）
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        前向传播：将输入图像通过网络，输出类别 logits

        :param x: 输入张量 (B, 1, 28, 28)
        :return: 类别 logits (B, 10)，未经 softmax
        """
        return self.net(x)


def main():
    """训练 CNN 3 个 epoch，输出测试集准确率。"""
    # ---- 数据准备 ----
    # ToTensor 将 PIL 图像或 numpy 数组转为 PyTorch 张量，并自动归一化到 [0, 1]
    transform = transforms.ToTensor()
    # 下载/加载 MNIST 训练集（60000 张）和测试集（10000 张）
    # download=True 表示本地不存在时自动从网上下载
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    # DataLoader 将数据集封装为可迭代的批量加载器
    # shuffle=True 每个 epoch 随机打乱训练数据，避免模型记住顺序
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)  # 测试集不需要打乱

    # ---- 模型、损失函数、优化器 ----
    # 自动选择 GPU（如果有）或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)  # 将模型移到指定设备
    # CrossEntropyLoss 内部包含 softmax + 负对数似然，适用于多分类
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器：自适应学习率算法，结合了 Momentum 和 RMSProp 的优点
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---- 训练循环 ----
    for epoch in range(3):
        model.train()  # 设置为训练模式（启用 dropout、batchnorm 的训练行为）
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # 将数据移到 GPU/CPU
            optimizer.zero_grad()       # 清空上一步的梯度（PyTorch 默认累积梯度）
            loss = criterion(model(X), y)  # 前向传播 + 计算损失
            loss.backward()             # 反向传播：自动计算所有参数的梯度
            optimizer.step()            # 根据梯度更新参数
            total_loss += loss.item()
        print(f"epoch {epoch+1}, loss={total_loss/len(train_loader):.4f}")

    # ---- 测试评估 ----
    model.eval()  # 设置为评估模式（关闭 dropout、batchnorm 使用运行统计量）
    preds, trues = [], []
    # torch.no_grad() 上下文管理器：禁用梯度计算，节省内存和计算
    with torch.no_grad():
        for X, y in test_loader:
            # 前向传播得到 logits，argmax 取概率最大的类别索引
            pred = model(X.to(device)).argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y.numpy())
    # 计算整体准确率
    print("CNN MNIST test acc:", accuracy_score(trues, preds))


if __name__ == "__main__":
    main()
