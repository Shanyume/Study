#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN（卷积神经网络）MNIST 分类
==============================

CNN 通过卷积核提取局部特征，池化降低空间分辨率，
是图像分类的经典架构。本脚本用两层卷积 + 全连接实现 MNIST 手写数字识别。

网络结构：
    - Conv(1→32) → ReLU → MaxPool(2)
    - Conv(32→64) → ReLU → MaxPool(2)
    - Flatten → Linear(3136→128) → ReLU → Linear(128→10)

依赖：PyTorch、torchvision、scikit-learn
运行：python cnn_mnist.py（首次运行会下载 MNIST 数据集）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score


class SimpleCNN(nn.Module):
    """简洁 CNN：两层卷积 + 两层全连接。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # 1 通道 → 32 通道，3×3 卷积核，padding 保持尺寸
            nn.ReLU(),                         # 非线性激活
            nn.MaxPool2d(2),                   # 28×28 → 14×14
            nn.Conv2d(32, 64, 3, padding=1),   # 32 → 64 通道
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14×14 → 7×7
            nn.Flatten(),                      # (B, 64, 7, 7) → (B, 3136)
            nn.Linear(64*7*7, 128),            # 3136 → 128
            nn.ReLU(),
            nn.Linear(128, 10)                 # 128 → 10 类
        )

    def forward(self, x):
        """前向传播。"""
        return self.net(x)


def main():
    """训练 CNN 3 个 epoch，输出测试集准确率。"""
    # 数据预处理：只做 ToTensor
    transform = transforms.ToTensor()
    # 下载/加载 MNIST 训练集和测试集
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    # 自动选择 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练 3 个 epoch
    for epoch in range(3):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()       # 清空梯度
            loss = criterion(model(X), y)  # 前向 + 计算损失
            loss.backward()             # 反向传播
            optimizer.step()            # 更新参数
            total_loss += loss.item()
        print(f"epoch {epoch+1}, loss={total_loss/len(train_loader):.4f}")

    # 测试
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X.to(device)).argmax(dim=1).cpu().numpy()  # # 取概率最大的类别索引
            preds.extend(pred)
            trues.extend(y.numpy())
    print("CNN MNIST test acc:", accuracy_score(trues, preds))


if __name__ == "__main__":
    main()
