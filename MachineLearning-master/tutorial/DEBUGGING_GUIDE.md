# 常见问题排查指南

学习代码时遇到的常见问题和解决方法。

## 1. NumPy 2.x 兼容问题

### np.float / np.int 被移除
```python
# 旧代码
np.float, np.int
# 新代码
float, int, np.float64, np.int64
```

### np.mat 被移除
```python
# 旧代码
np.mat(X)
# 新代码
np.asmatrix(X)
```

## 2. Python 3 兼容问题

### print 语句
```python
# 旧
print "hello"
# 新
print("hello")
```

### iteritems / iterkeys
```python
# 旧
dict.iteritems()
# 新
dict.items()
```

### pickle
```python
# 旧
import cPickle
cPickle.load(f)
# 新
import pickle
pickle.load(f, encoding='latin1')
```

## 3. scikit-learn 版本差异

### sklearn.lda 已移除
```python
# 旧
from sklearn import lda
lda.LDA()
# 新
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LinearDiscriminantAnalysis()
```

### 参数必须用关键字
```python
# 旧（位置参数）
Isomap(n_neighbors, n_components=2)
# 新（关键字参数）
Isomap(n_neighbors=n_neighbors, n_components=2)
```

## 4. PyTorch 常见问题

### MNIST 下载失败
网络问题时，手动下载数据到 `./data`：
```bash
mkdir -p data/MNIST/raw
# 下载 train-images-idx3-ubyte.gz 等文件
```

### CUDA 不可用
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 5. matplotlib 中文显示

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

## 6. 无界面环境运行

```bash
MPLBACKEND=Agg python script.py
```

## 7. Git 提交常见问题

### index.lock 残留
```bash
ls -la .git/index.lock
# 如果存在，删除：
rm .git/index.lock
```

### 权限问题
```bash
ls -ld .git
# 确保当前用户有写权限
```

## 8. 学习代码的调试技巧

### 打印形状
```python
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

### 检查 NaN / Inf
```python
print(np.isnan(X).sum(), np.isinf(X).sum())
```

### 固定随机种子
```python
np.random.seed(42)
# 或
rng = np.random.RandomState(42)
```

### 小数据集测试
先用 20-50 个样本测试代码逻辑，再跑全量数据。
