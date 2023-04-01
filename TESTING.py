import numpy as np

from single_link import SingleLink
from average_linkage import AverageLinkage
from kmedoids import KMedoidsClustering
from distance import Parser, euclidean_distance
from util import plot_umap


np.set_printoptions(threshold=1000000)
parser = Parser()
dm = parser.get_distance_matrix("cnnhealth.txt", euclidean_distance, "distance_matrix.npy")
bow = parser.get_bag_of_words("cnnhealth.txt", "bag_of_words.npy")

km = KMedoidsClustering(num_clusters=8)
km_clusters = km.cluster(dm)
plot_umap(bow, km_clusters)

sl = SingleLink(num_clusters=8)
sl_clusters = sl.cluster(dm)
plot_umap(bow, sl_clusters)

al = AverageLinkage(num_clusters=8)
al_clusters = al.cluster(dm)
plot_umap(bow, al_clusters)