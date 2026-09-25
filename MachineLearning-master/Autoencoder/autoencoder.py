#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder（自编码器）MNIST 降维与重构
========================================

自编码器是一种无监督神经网络，由编码器和解码器组成：
    - 编码器：把高维输入压缩到低维潜变量 z
    - 解码器：从 z 重构出尽量接近原始输入的输出
训练目标是最小化重构误差（MSE），不使用标签。

网络结构：
    - Encoder: 28×28 → 128 → 32
    - Decoder: 32 → 128 → 28×28

依赖：PyTorch、torchvision
运行：python autoencoder.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Autoencoder(nn.Module):
    """全连接自编码器，28×28 图像压缩到 latent_dim 维潜变量再重构。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        # 编码器：Flatten 后逐层降维到潜变量维度
        self.encoder = nn.Sequential(
            nn.Flatten(),               # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(28*28, 128),      # 784 -> 128
            nn.ReLU(),                  # 非线性激活
            nn.Linear(128, latent_dim), # 128 -> 32（潜变量维度）
            nn.ReLU()                   # 潜变量也用 ReLU 保持非负
        )
        # 解码器：从潜变量逐层升维回原始尺寸
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), # 32 -> 128
            nn.ReLU(),
            nn.Linear(128, 28*28),      # 128 -> 784
            nn.Sigmoid(),               # 输出范围 [0, 1]，与像素值匹配
            nn.Unflatten(1, (1, 28, 28)) # (B, 784) -> (B, 1, 28, 28)
        )

    def forward(self, x):
        """前向传播：编码到潜变量，再解码重构。"""
        z = self.encoder(x)   # 潜变量 z，形状 (B, 32)
        return self.decoder(z) # 重构图像，形状 (B, 1, 28, 28)


def main():
    """训练自编码器 5 个 epoch，输出每轮平均重构损失。"""
    # 数据预处理：只做 ToTensor，把像素从 [0,255] 缩放到 [0,1]
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    # 自动选择 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(latent_dim=32).to(device)

    # MSE 损失：衡量重构图像与原始图像的像素级差异
    criterion = nn.MSELoss()
    # Adam 优化器，学习率 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环：5 个 epoch
    for epoch in range(5):
        model.train()
        total_loss = 0
        for X, _ in train_loader:  # 无监督，不需要标签
            X = X.to(device)
            optimizer.zero_grad()          # 清空梯度
            recon = model(X)               # 前向传播得到重构图像
            loss = criterion(recon, X)     # 计算重构损失
            loss.backward()                # 反向传播
            optimizer.step()               # 更新参数
            total_loss += loss.item()
        # 输出本轮平均损失
        print(f"epoch {epoch+1}, recon loss={total_loss/len(train_loader):.6f}")


if __name__ == "__main__":
    main()
