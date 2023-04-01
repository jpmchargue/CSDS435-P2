from clustering import Clustering
from sklearn_extra.cluster import KMedoids
import numpy as np

class KMedoidsClustering(Clustering):
    def __init__(self, num_clusters: int=8):
        self.num_clusters = num_clusters
        self.model = KMedoids(
            n_clusters=self.num_clusters, 
            metric="precomputed"
        )

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with K-Medoids...")
        clustering = self.model.fit(distance_matrix)
        return clustering.labels_