# -*- coding: utf-8 -*-
"""
逻辑回归（Logistic Regression）+ 随机梯度下降（SGD）训练 MNIST
================================================================

本文件实现了一个基于 Theano 框架的逻辑回归分类器，
并使用小批量随机梯度下降（Minibatch SGD）在 MNIST 手写数字数据集上进行训练和评估。

算法原理：
    1. 逻辑回归本质上是多分类的 softmax 回归：
       - 输入 x（784 维 = 28×28 展平的 MNIST 图像）
       - 线性变换：z = xW + b，其中 W 为 (784, 10)，b 为 (10,)
       - softmax 激活：p(y=k|x) = exp(z_k) / Σ_j exp(z_j)
       - 预测类别：argmax_k p(y=k|x)
    2. 损失函数为负对数似然（等价于交叉熵）：
       L = -(1/m) Σ_i log p(y_i | x_i)
    3. 优化算法为 Minibatch SGD：
       - 每次取 batch_size 个样本计算梯度
       - 参数更新：W ← W - lr × ∂L/∂W，b ← b - lr × ∂L/∂b
    4. 早停策略（Early Stopping）：
       - 定期在验证集上评估，若验证误差不再改善则提前终止

数据集：MNIST 手写数字（60000 训练 / 10000 测试，10 类，28×28 灰度图）

依赖：numpy、theano
运行：python logistic_sgd.py（首次运行会自动下载 MNIST 数据集）
"""

__docformat__ = 'restructedtext en'

import pickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """
    逻辑回归分类器（多分类 softmax 回归）

    模型结构：
        输入 x (m, n_in) → 线性变换 z = xW + b → softmax → 概率分布 p(y|x)
        其中 W 为 (n_in, n_out) 的权重矩阵，b 为 (n_out,) 的偏置向量

    属性：
        W: theano.shared，权重矩阵，初始化为零矩阵
        b: theano.shared，偏置向量，初始化为零向量
        p_y_given_x: 符号表达式，每个样本属于各类别的概率分布
        y_pred: 符号表达式，预测的类别标签（概率最大值的下标）
        params: [W, b]，模型的可训练参数列表
    """

    def __init__(self, input, n_in, n_out):
        """
        初始化逻辑回归模型

        :param input: Theano 符号变量，表示输入数据（m, n_in）
        :param n_in: 输入特征维度（MNIST 为 28×28 = 784）
        :param n_out: 输出类别数（MNIST 为 10）
        """
        # 初始化权重矩阵 W 为零矩阵 (n_in, n_out)
        # 使用 theano.shared 使其成为共享变量，可在 GPU 上存储和加速计算
        # borrow=True 表示 Theano 可以直接引用现有内存而不复制，节省内存
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # 初始化偏置向量 b 为零向量 (n_out,)
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

    if os.path.exists('params'):
        f=open('params')
        self.W.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)


        # 计算每个样本属于各类别的概率分布
        # T.dot(input, self.W) 计算线性变换 xW，形状为 (m, n_out)
        # 加上偏置 b 后（广播机制，b 自动加到每一行），送入 softmax
        # softmax 按行归一化，使每行和为 1，表示概率分布
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # 按行取概率最大值的下标作为预测类别
        # axis=1 表示沿列方向（即对每个样本）取 argmax
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        计算负对数似然损失（交叉熵损失）

        公式：L = -(1/m) Σ_i log p(y_i | x_i)
        其中 p(y_i|x_i) 是模型对样本 i 的真实类别 y_i 所预测的概率

        :param y: 真实标签，形状 (m,)，每个元素为 0~n_out-1 的整数
        :return: 标量损失值
        """
        # T.log(self.p_y_given_x) 对所有类别的概率取对数
        # [T.arange(y.shape[0]), y] 用花式索引取出每个样本真实类别对应的 log 概率
        # T.mean 取均值，得到平均负对数似然
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        计算分类错误率（zero-one loss）

        预测类别与真实类别不一致的样本占比

        :param y: 真实标签，形状 (m,)
        :return: 标量错误率（0~1 之间）
        """
        # 检查预测值和真实值的维度是否一致
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # 仅支持整数类型的标签
        if y.dtype.startswith('int'):
            # T.neq 逐元素比较不相等，返回 0/1；T.mean 求均值即错误率
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    """
    加载 MNIST 数据集

    数据集以 pickle 格式压缩存储（.pkl.gz），包含三个子集：
        - train_set: 训练集（50000 样本）
        - valid_set: 验证集（10000 样本）
        - test_set: 测试集（10000 样本）

    每个子集是 (data, label) 的元组：
        - data: (n_samples, 784) 的浮点数组，每行是一张展平的 28×28 图像
        - label: (n_samples,) 的整数数组，值为 0~9

    加载后会将数据转换为 Theano 共享变量，以便在 GPU 上加速计算。

    :param dataset: 数据集文件路径（如 'mnist.pkl.gz'）
    :return: [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    """
    # 解析数据集路径：如果当前目录下找不到，尝试在 ../data/ 目录下查找
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    # 如果本地仍找不到且是 MNIST 数据集，则从网上自动下载
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib.request, urllib.parse, urllib.error
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # 用 gzip 解压并用 pickle 加载数据
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        """
        将数据集加载到 Theano 共享变量中

        使用共享变量的原因：
            - Theano 可以将共享变量复制到 GPU 显存中，避免每次 minibatch 都从 CPU 拷贝数据
            - GPU 数据类型必须是 float，所以标签先存为 float 再 cast 为 int32

        :param data_xy: (data_x, data_y) 的元组
        :param borrow: 是否允许 Theano 直接引用内存而不复制
        :return: (shared_x, shared_y)，其中 shared_y 已 cast 为 int32
        """
        data_x, data_y = data_xy
        # 将特征数据转为共享变量（float 类型，兼容 GPU）
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # 标签也先存为 float 共享变量，后续 cast 为 int32
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    # 分别将测试集、验证集、训练集转为共享变量
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def save_params(param1,param2):
    """
    保存模型参数到文件

    将训练好的权重 W 和偏置 b 序列化保存到 'params' 文件中，
    下次训练时 LogisticRegression.__init__ 会自动读取该文件来初始化参数。

    :param param1: 权重矩阵 W 的 theano.shared 变量
    :param param2: 偏置向量 b 的 theano.shared 变量
    """
    import pickle
    write_file = open('params', 'wb') 
    # 用 get_value() 从共享变量中取出 numpy 数组再序列化
    pickle.dump(param1.get_value(borrow=True), write_file, -1)
    pickle.dump(param2.get_value(borrow=True), write_file, -1)
    write_file.close()	

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=3,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    用 SGD 训练逻辑回归模型并在 MNIST 上评估

    训练流程：
        1. 加载数据并划分为训练/验证/测试集
        2. 构建逻辑回归模型（Theano 计算图）
        3. 定义代价函数、梯度计算和参数更新规则
        4. 编译 Theano 函数（train_model / validate_model / test_model）
        5. 训练循环：逐 batch 训练，定期在验证集上评估，使用早停策略

    早停策略（Early Stopping）：
        - patience: 最多允许训练多少个 batch 而不改善验证误差
        - patience_increase: 当验证误差显著改善时，放大 patience
        - improvement_threshold: 验证误差需改善到此比例以下才算"显著改善"

    :param learning_rate: 学习率（梯度前的系数），默认 0.13
    :param n_epochs: 最大训练轮数，默认 3
    :param dataset: 数据集文件路径
    :param batch_size: 每个 minibatch 的样本数，默认 600
    """
    # 加载数据
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算各数据集的 batch 数量
    # 样本总数 / batch_size = batch 数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    print('... building the model')

    # index 是标量符号变量，表示当前 batch 的下标（0, 1, 2, ...）
    index = T.lscalar()  # index to a [mini]batch

    # x 是矩阵符号变量，表示输入图像数据（batch_size, 784）
    # 每张 MNIST 图像是 28×28 的灰度图，展平为 784 维向量
    x = T.matrix('x')  # data, presented as rasterized images
    # y 是整数向量符号变量，表示标签（batch_size,），每个元素为 0~9
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # 构建逻辑回归分类器
    # n_in = 28*28 = 784（输入特征维度）
    # n_out = 10（输出类别数，对应 0~9 十个数字）
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # 代价函数：负对数似然（交叉熵损失）
    cost = classifier.negative_log_likelihood(y)


    # 编译测试函数：给定 batch 下标 index，在测试集上计算分类错误率
    # givens 字典将符号变量 x, y 替换为具体数据切片
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 编译验证函数：与 test_model 结构相同，只是数据源换为验证集
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 计算代价函数对各参数的梯度（自动微分）
    # g_W = ∂cost/∂W，g_b = ∂cost/∂b
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # 参数更新规则：W ← W - lr × g_W，b ← b - lr × g_b
    # 这是 SGD 的核心：沿梯度反方向更新参数以最小化损失
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]


    # 编译训练函数：给定 batch 下标，计算损失并更新参数
    # updates 参数使每次调用 train_model 时自动执行参数更新
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # 训练模型     #
    ###############
    print('... training the model')
    # 早停参数
    patience = 5000  # 最多容忍多少个 batch 不改善就停止
    patience_increase = 2  # 验证误差显著改善时，patience 放大到此倍数
    improvement_threshold = 0.995  # 验证误差需改善到此比例以下才算显著改善
    validation_frequency = min(n_train_batches, patience / 2)
                                  # 每隔多少个 batch 在验证集上评估一次
                                  # 这里保证每个 epoch 至少评估一次

    best_validation_loss = numpy.inf  # 历史最优验证损失，初始为无穷大
    test_score = 0.  # 最优模型对应的测试集误差
    start_time = time.clock()

    done_looping = False
    epoch = 0
    # 主训练循环：外层遍历 epoch，内层遍历每个 batch
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            # 训练一个 batch：前向传播计算损失，反向传播计算梯度，更新参数
            minibatch_avg_cost = train_model(minibatch_index)
            # 累计迭代次数（跨 epoch）
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # 每隔 validation_frequency 个 batch 在验证集上评估
            if (iter + 1) % validation_frequency == 0:
                # 在所有验证 batch 上计算错误率并取均值
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print((
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                ))

                # 如果当前验证误差是历史最优，则更新并保存参数
                if this_validation_loss < best_validation_loss:
                    # 如果改善幅度超过阈值，则放大 patience（延长训练）
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # 保存当前最优参数到文件
                    save_params(classifier.W,classifier.b)

                    # 在测试集上评估当前最优模型
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print((
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    ))

            # 如果累计迭代次数超过 patience，则提前终止训练
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print((
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    ))
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

if __name__ == '__main__':
    sgd_optimization_mnist()
