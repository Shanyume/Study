# Decision Tree

- 实现：[id3_c45.py](id3_c45.py)
- 树可视化：[treePlotter.py](treePlotter.py)
- 支持算法：ID3、C4.5
- 依赖：NumPy、Matplotlib

## 快速测试

```python
from id3_c45 import DecisionTree

X = [[1, 2, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 0, 0, 1],
     [2, 1, 1, 0, 1],
     [1, 1, 0, 1, 1]]
y = ['yes', 'yes', 'no', 'no', 'no']

clf = DecisionTree(mode='ID3')
clf.fit(X, y)
print(clf.predict(X))
```

## 已知限制

- 假设特征为离散值，未处理连续特征阈值分裂。
- 未加入 CART。
- 预测时若测试样本特征值未在训练集中出现，可能遇到 KeyError。
