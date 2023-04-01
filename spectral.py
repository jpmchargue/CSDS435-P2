from clustering import Clustering
from sklearn.cluster import SpectralClustering
import numpy as np


class SpecClustering(Clustering):
    def __init__(self, num_clusters: int = 8):
        self.num_clusters = num_clusters
        self.model = SpectralClustering(
            n_clusters=self.num_clusters, affinity="precomputed"
        )

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with spectral clustering...")
        clustering = self.model.fit(np.max(distance_matrix) - distance_matrix)
        return clustering.labels_
