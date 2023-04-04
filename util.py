import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import namedtuple
import math


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


Clusters = namedtuple("Clusters", "clusters name d_name eval s_matrix")


def perform_clustering(model, name, d_name, s_matrix, d_matrix, sim=False) -> Clusters:
    d_name += " Similarity" if sim else " Distance"

    print(name, "with", d_name)

    clusters = model.cluster(s_matrix if sim else d_matrix)
    ev = evaluate(d_matrix, s_matrix, clusters)

    unique, counts = np.unique(clusters, return_counts=True)
    plt.title(f"{name} Cluster Distributions with {d_name}")
    plt.xlabel("Cluster")
    plt.ylabel("Size")
    plt.bar(unique, counts)
    plt.show()

    return Clusters(
        clusters=clusters, name=name, d_name=d_name, eval=ev, s_matrix=s_matrix
    )


def perform_comparison(c1: Clusters, c2: Clusters):
    print("Comparison:")

    plt.figure().set_figwidth(12)
    axs = [plt.subplot2grid((1, 2), (0, i)) for i in range(2)]
    for i, c in enumerate([c1, c2]):
        axs[i].set_title(f"{c.name} with {c.d_name}")
        im = plot_cluster_sim(axs[i], c.s_matrix, c.clusters)
        print("Classes =", c.name, "width", c.d_name)
        comp = compare(c.clusters, [c2, c1][i].clusters)
        plt.colorbar(
            im, cax=make_axes_locatable(axs[i]).append_axes("right", size="5%", pad=0.1)
        )
    plt.show()


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


Comparison = namedtuple("Comparison", "entropy purity")


def entropy(p):
    return 0 if p == 0 else -p * math.log2(p)


def compare(cluster1, cluster2) -> Comparison:
    class_idxs = [np.where(cluster1 == i)[0] for i in np.unique(cluster1)]
    cluster_idxs = [np.where(cluster2 == i)[0] for i in np.unique(cluster2)]
    p_s = [
        [len(np.intersect1d(cla, clu)) / len(clu) for cla in class_idxs]
        for clu in cluster_idxs
    ]
    clu_ents = [sum([entropy(p_ij) for p_ij in row]) for row in p_s]
    clu_purs = [max(row) for row in p_s]

    comp = Comparison(
        entropy=sum(
            [ent * len(cla) / len(cluster1) for cla, ent in zip(class_idxs, clu_ents)]
        ),
        purity=sum(
            [pur * len(cla) / len(cluster1) for cla, pur in zip(class_idxs, clu_purs)]
        ),
    )

    print("Entropy:", comp.entropy)
    print("Purity:", comp.purity)

    return comp
