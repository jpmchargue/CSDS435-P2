import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import namedtuple


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


def plot_cluster_sim(ax, sim_matrix, clusters):
    idxs = np.argsort(clusters)
    sim_matrix = sim_matrix[idxs]
    sim_matrix[:] = sim_matrix[:, idxs]
    min_v = np.min(sim_matrix)
    max_v = np.max(sim_matrix)
    sim_matrix[:] = (sim_matrix - min_v) / (max_v - min_v)
    return ax.imshow(sim_matrix, interpolation="nearest")


Evaluation = namedtuple("Evaluation", "cohesion separation silhouette")


def weighted_avg(vals, classes):
    N = sum([len(c) for c in classes])
    return sum([v * len(c) / N for v, c in zip(vals, classes)])


def evaluate(d_matrix, sim_matrix, clusters) -> Evaluation:
    unique = np.unique(clusters)
    cluster_idxs = [np.where(clusters == i)[0] for i in unique]
    cluster_sims = [sim_matrix[idxs, :][:, idxs] for idxs in cluster_idxs]
    inv_cluster_sims = [
        np.delete(sim_matrix[idxs, :], idxs, axis=1) for idxs in cluster_idxs
    ]

    a_s = np.array(
        [np.average(d_matrix[i][cluster_idxs[c]]) for i, c in enumerate(clusters)]
    )
    b_s = np.array(
        [
            np.average(np.delete(d_matrix[i], cluster_idxs[c]))
            for i, c in enumerate(clusters)
        ]
    )
    max_s = np.array([max(a_i, b_i) for a_i, b_i in zip(a_s, b_s)])
    sils = (b_s - a_s) / max_s

    ev = Evaluation(
        cohesion=[np.average(sims.flatten()) for sims in cluster_sims],
        separation=[np.average(sims.flatten()) for sims in inv_cluster_sims],
        silhouette=[np.average(sils[idxs]) for idxs in cluster_idxs],
    )

    print("Cohesion:", ev.cohesion)
    print("Avg Cohesion:", weighted_avg(ev.cohesion, cluster_idxs))
    print("Separation:", ev.separation)
    print("Avg Separation:", weighted_avg(ev.separation, cluster_idxs))
    print("Silhouette:", ev.silhouette)
    print("Avg Silhouette:", weighted_avg(ev.silhouette, cluster_idxs))

    return ev
