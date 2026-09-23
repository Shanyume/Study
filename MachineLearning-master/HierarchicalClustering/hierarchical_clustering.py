"""层次聚类（AgglomerativeClustering）示例，不同 linkage 对比。"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

BASE = __import__('pathlib').Path(__file__).resolve().parent

def main():
    X, y = load_iris(return_X_y=True)
    for linkage_method in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(n_clusters=3, linkage=linkage_method)
        labels = model.fit_predict(X)
        print(f"linkage={linkage_method:>8}, ARI={adjusted_rand_score(y, labels):.3f}")
    # 绘制树状图
    Z = linkage(X, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.xlabel("Sample index"); plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(BASE / "dendrogram.png", dpi=120)
    print("dendrogram saved:", BASE / "dendrogram.png")

if __name__ == "__main__":
    main()
