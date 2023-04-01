from clustering import Clustering
from sklearn.cluster import KMeans


class KMeansClustering(Clustering):
    def __init__(self, num_clusters: int = 8):
        self.num_clusters = num_clusters
        self.model = KMeans(
            n_clusters=self.num_clusters,
        )
