from clustering import Clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class AverageLinkage(Clustering):
    def __init__(self, num_clusters: int=8):
        self.num_clusters = num_clusters
        self.model = AgglomerativeClustering(
            n_clusters=self.num_clusters, 
            metric="precomputed", 
            linkage="average"
        )

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with average linkage...")
        clustering = self.model.fit(distance_matrix)
        return clustering.labels_