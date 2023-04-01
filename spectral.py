from clustering import Clustering
from sklearn.cluster import SpectralClustering
import numpy as np


class SpecClustering(Clustering):
    def __init__(self, num_clusters: int = 8):
        self.num_clusters = num_clusters
        self.model = SpectralClustering(
            n_clusters=self.num_clusters, affinity="precomputed"
        )

    def cluster(self, affinity_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with spectral clustering...")
        clustering = self.model.fit(affinity_matrix)
        return clustering.labels_
