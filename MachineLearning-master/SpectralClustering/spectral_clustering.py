"""谱聚类示例：Iris 数据集。"""
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

def main():
    X, y = load_iris(return_X_y=True)
    model = SpectralClustering(n_clusters=3, affinity="nearest_neighbors", n_neighbors=10, random_state=42)
    labels = model.fit_predict(X)
    print("ARI:", adjusted_rand_score(y, labels))

if __name__ == "__main__":
    main()
