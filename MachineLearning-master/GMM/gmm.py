"""GMM 高斯混合模型（EM 算法）与 KMeans 对比。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, seed=42):
        self.k = n_components; self.max_iter = max_iter; self.tol = tol; self.seed = seed
    def fit(self, X):
        X = np.asarray(X); rng = np.random.RandomState(self.seed)
        n, d = X.shape
        # 初始化：随机选 k 个点作为均值，协方差为单位阵
        self.mu = X[rng.choice(n, self.k, replace=False)]
        self.cov = np.array([np.eye(d) for _ in range(self.k)])
        self.pi = np.ones(self.k) / self.k
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step
            resp = np.zeros((n, self.k))
            for i in range(self.k):
                resp[:, i] = self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i])
            resp /= resp.sum(axis=1, keepdims=True) + 1e-12
            # M-step
            Nk = resp.sum(axis=0)
            self.pi = Nk / n
            self.mu = (resp.T @ X) / Nk[:, None]
            for i in range(self.k):
                diff = X - self.mu[i]
                self.cov[i] = (resp[:, i][:, None] * diff).T @ diff / Nk[i] + 1e-6*np.eye(d)
            # log-likelihood
            densities = np.array([
                self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i])
                for i in range(self.k)])
            ll = np.sum(np.log(densities.sum(axis=0) + 1e-12))
            if abs(ll - prev_ll) < self.tol: break
            prev_ll = ll
        self.resp = resp
        return self
    def predict(self, X):
        X = np.asarray(X); probs = np.array([
            self.pi[i] * multivariate_normal.pdf(X, self.mu[i], self.cov[i]) for i in range(self.k)])
        return probs.argmax(axis=0)

def best_label_mapping(y_true, y_pred, k):
    """聚类标签和真实标签没有固定对应，做最优映射后计算准确率。"""
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in y_pred])
        acc = accuracy_score(y_true, mapped)
        best_acc = max(best_acc, acc)
    return best_acc

if __name__ == "__main__":
    from KMeans.kmeans import KMeans
    X, y = load_iris(return_X_y=True)
    gmm = GMM(n_components=3).fit(X)
    print("GMM acc:", best_label_mapping(y, gmm.predict(X), 3))
    km = KMeans(n_clusters=3, max_iter=50); km.fit(X)
    print("KMeans acc:", best_label_mapping(y, km.labels.astype(int), 3))
