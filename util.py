import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

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