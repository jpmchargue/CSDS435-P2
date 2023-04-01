import numpy as np
from clustering import Clustering
from sklearn.cluster import DBSCAN


class DBScanClustering(Clustering):
    def __init__(self, eps: int = 1):
        self.eps = eps
        self.model = DBSCAN(metric="precomputed", eps=eps)

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        print(f"Clustering with dbscan(eps={self.eps})...")
        clustering = self.model.fit(distance_matrix)
        return clustering.labels_
