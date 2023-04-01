import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_umap(bow, labels):
    color_ref = [
        "slategray",
        "r",
        "orange",
        "y",
        "g",
        "aqua",
        "b",
        "indigo",
        "m",
        "black",
    ]
    colors = [color_ref[l] for l in labels]

    scaled = StandardScaler().fit_transform(bow)
    umapped = umap.UMAP().fit_transform(scaled)

    print("Plotting...")
    plt.scatter(umapped[:, 0], umapped[:, 1], c=colors)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP Projection")
    plt.show()


def plot_dists(ax, dist_matrix):
    ax.hist(dist_matrix[np.triu_indices(len(dist_matrix), k=1)], bins=100)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
