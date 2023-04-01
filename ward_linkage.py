from turtle import distance
from clustering import Clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class WardLinkage(Clustering):
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters

    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        return AgglomerativeClustering(
            n_clusters=self.num_clusters, 
            metric="precomputed", 
            linkage="ward"
        ).fit(distance_matrix)