#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU 导数"""
    return (x > 0).astype(float)


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Sigmoid 导数"""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """Softmax 函数（用于多分类输出）"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MLP:
    """
    多层感知机（Multi-Layer Perceptron）
    支持自定义隐藏层结构和激活函数
    """

    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01, n_epochs=1000):
        """
        :param layer_sizes: 网络结构，如 [784, 128, 64, 10] 表示输入 784，两个隐藏层 128/64，输出 10
        :param activation: 隐藏层激活函数，'relu' 或 'sigmoid'
        :param learning_rate: 学习率
        :param n_epochs: 训练轮数
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = []
        self.biases = []

        # 设置激活函数
        if activation == 'relu':
            self.activation = relu
            self.activation_deriv = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_derivative
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 初始化权重和偏置（Xavier 初始化）
        self._initialize_weights()

    def _initialize_weights(self):
        """
        使用 Xavier 初始化方法初始化权重
        权重从均匀分布 U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))) 中采样
        """
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            # Xavier 初始化
            limit = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据，形状 (n_samples, n_features)
        :return: 输出和各层激活值
        """
        activations = [X]
        z_values = []

        current = X
        for i in range(len(self.weights) - 1):
            # 隐藏层：线性变换 + 激活函数
            z = current @ self.weights[i] + self.biases[i]
            z_values.append(z)
            current = self.activation(z)
            activations.append(current)

        # 输出层：线性变换 + softmax（多分类）
        z = current @ self.weights[-1] + self.biases[-1]
        z_values.append(z)
        output = softmax(z)
        activations.append(output)

        return activations, z_values

    def backward(self, activations, z_values, y_true):
        """
        反向传播，计算梯度
        :param activations: 各层激活值列表
        :param z_values: 各层线性输出列表
        :param y_true: 真实标签（one-hot 编码）
        """
        n_samples = y_true.shape[0]
        n_layers = len(self.weights)

        # 输出层梯度（softmax + cross-entropy 的梯度简化为 output - y_true）
        delta = activations[-1] - y_true

        dw_list = []
        db_list = []

        # 从后向前计算梯度
        for i in range(n_layers - 1, -1, -1):
            # 计算权重和偏置的梯度
            dw = (activations[i].T @ delta) / n_samples
            db = np.sum(delta, axis=0, keepdims=True) / n_samples

            dw_list.insert(0, dw)
            db_list.insert(0, db)

            # 传播到前一层（如果不是第一层）
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_deriv(z_values[i - 1])

        # 更新参数
        for i in range(n_layers):
            self.weights[i] -= self.learning_rate * dw_list[i]
            self.biases[i] -= self.learning_rate * db_list[i]

    def fit(self, X, y):
        """
        训练模型
        :param X: 训练特征，形状 (n_samples, n_features)
        :param y: 训练标签，形状 (n_samples,) 或 one-hot 编码
        """
        # 将标签转换为 one-hot 编码
        if y.ndim == 1:
            n_classes = self.layer_sizes[-1]
            y_onehot = np.zeros((len(y), n_classes))
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
        else:
            y_onehot = y

        self.loss_history = []

        for epoch in range(self.n_epochs):
            # 前向传播
            activations, z_values = self.forward(X)

            # 计算交叉熵损失
            loss = -np.mean(np.sum(y_onehot * np.log(activations[-1] + 1e-10), axis=1))
            self.loss_history.append(loss)

            # 反向传播
            self.backward(activations, z_values, y_onehot)

            # 每 100 轮打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        预测类别
        :param X: 输入数据
        :return: 预测类别标签
        """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)  # # 取输出层激活最大的神经元索引作为预测类别

    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    演示 MLP 分类
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("=" * 50)
    print("多层感知机 (MLP) 分类演示")
    print("=" * 50)

    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建 MLP：20 输入 -> 64 隐藏 -> 32 隐藏 -> 3 输出
    mlp = MLP(layer_sizes=[20, 64, 32, 3],
              activation='relu',
              learning_rate=0.01,
              n_epochs=500)

    print("\n训练 MLP...")
    mlp.fit(X_train, y_train)

    # 评估
    train_acc = mlp.score(X_train, y_train)
    test_acc = mlp.score(X_test, y_test)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 绘制损失曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mlp_loss_curve.png', dpi=150)
    print("\nPlot saved: mlp_loss_curve.png")
    plt.close()


if __name__ == '__main__':
    demo()
