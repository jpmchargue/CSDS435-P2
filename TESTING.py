import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from single_link import SingleLink
from average_linkage import AverageLinkage
from kmedoids import KMedoidsClustering
from distance import Parser, euclidean_distance


def plot_umap(bow, labels):
    color_ref = ["slategray", 'r', "orange", 'y', 'g', "aqua", 'b', "indigo", 'm', "black"]
    colors = [color_ref[l] for l in labels]

    scaled = StandardScaler().fit_transform(bow)
    umapped = umap.UMAP().fit_transform(scaled)

    print("Plotting...")
    plt.scatter(
        umapped[:, 0],
        umapped[:, 1],
        c=colors
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP Projection')
    plt.show()

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