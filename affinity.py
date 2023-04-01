from clustering import Clustering
from sklearn.cluster import AffinityPropagation
import numpy as np


class AffinityClustering(Clustering):
    def __init__(self):
        self.model = AffinityPropagation(affinity="precomputed", damping=0.9)

    def cluster(self, affinity_matrix: np.ndarray) -> np.ndarray:
        print("Clustering with affinity propogation...")
        clustering = self.model.fit(affinity_matrix)
        return clustering.labels_
