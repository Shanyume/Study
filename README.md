# 深度学习与机器学习资料总览

这是一个个人学习工作区，包含笔记、课程代码和参考书。本仓库的目标是把零散资料组织成一条可执行的学习路线。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| [`dl学习资料/`](dl学习资料/README.md) | 主力笔记，覆盖 PyTorch、深度学习、CV、NLP、大模型与 Agent |
| [`MachineLearning_Python-master/`](MachineLearning_Python-master/readme.md) | 经典机器学习算法的 Python 实现 |
| [`MachineLearning-master/`](MachineLearning-master/README.md) | 另一套机器学习实现与说明 |
| [`deeplearning-models-master/`](deeplearning-models-master/README.md) | PyTorch / TensorFlow 模型与训练示例 |

根目录下还有两本 EPUB 电子书：《机器学习》和《深度学习》。

## 学习路线

建议按下面的顺序推进，不要一开始就在多个仓库之间来回切换。

1. **PyTorch 基础**
   - 位置：`dl学习资料/100_配置版本.ipynb` 到 `122_查看开源项目.ipynb`
   - 目标：能独立完成数据加载、模型训练、验证、保存和读取。
2. **深度学习基础**
   - 位置：`dl学习资料/200_深度学习介绍.ipynb` 到 `215_PyTorch神经网络基础.ipynb`
   - 目标：掌握张量操作、自动求导、线性回归、Softmax、MLP、正则化和优化。
3. **CV 与经典模型**
   - 位置：`dl学习资料/216_卷积层.ipynb` 到 `245_样式迁移.ipynb`
   - 目标：理解 CNN、ResNet、批量归一化、目标检测、语义分割和迁移学习。
4. **NLP 与序列模型**
   - 位置：`dl学习资料/246_序列模型.ipynb` 到 `265_BERT微调.ipynb`
   - 目标：理解 RNN、GRU、LSTM、Seq2Seq、注意力、Transformer 和 BERT。
5. **课程巩固**
   - 位置：`dl学习资料/300_配置版本.ipynb` 到 `354_课程5_第3周_作业题b_语音识别关键字.ipynb`
   - 目标：用吴恩达课程补齐理论基础，并完成配套作业。
6. **大模型与 Agent**
   - 位置：`dl学习资料/400_配置版本.ipynb` 到 `409_多轮对话.ipynb`
   - 目标：理解 RAG、向量数据库、API 调用、多轮对话和轻量手写实现。

## 环境准备

推荐使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

`requirements.txt` 只保留学习主线需要的核心依赖。某些 Notebook 可能还需要额外依赖，例如：

```bash
pip install transformers datasets accelerate langchain langchain-community faiss-cpu
```

如果你只需要运行基础机器学习脚本，不一定要安装完整的深度学习环境。

## 使用建议

- `dl学习资料/` 是主线，优先按编号顺序学习。
- `MachineLearning_Python-master/` 和 `MachineLearning-master/` 更适合当作对照实现，部分代码年代较早，遇到 API 不兼容时优先参考 `deeplearning-models-master/`。
- 建议每个主题学完后回到一个最小数据集上复现一次，而不是只看 Notebook。

