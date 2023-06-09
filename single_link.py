from turtle import distance
from clustering import Clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class SingleLink(Clustering):
    def __init__(self, num_clusters: int=8):
        self.num_clusters = num_clusters
        self.model = AgglomerativeClustering(
            n_clusters=self.num_clusters, 
            metric="precomputed", 
            linkage="single"
        )

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with single link...")
        clustering = self.model.fit(distance_matrix)
        return clustering.labels_