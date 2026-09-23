"""Mean Shift 聚类示例。"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score

def main():
    X, y = load_iris(return_X_y=True)
    bandwidth = estimate_bandwidth(X, quantile=0.3, random_state=42)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = model.fit_predict(X)
    print("n_clusters:", len(model.cluster_centers_))
    print("ARI:", adjusted_rand_score(y, labels))

if __name__ == "__main__":
    main()
