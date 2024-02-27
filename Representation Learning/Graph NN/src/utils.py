import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from umap import UMAP


def visualize_embeddings(z: torch.Tensor, y: torch.Tensor):
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    z2d_pca = PCA(n_components=2).fit_transform(z)
    z2d_umap = UMAP(n_components=2).fit_transform(z)

    sns.scatterplot(
        x=z2d_pca[:, 0],
        y=z2d_pca[:, 1],
        hue=y,
        palette="Set2",
        ax=axs[0],
    )
    axs[0].set(title="PCA")

    sns.scatterplot(
        x=z2d_umap[:, 0],
        y=z2d_umap[:, 1],
        hue=y,
        palette="Set2",
        ax=axs[1],
    )
    axs[1].set(title="UMAP")

    return fig
