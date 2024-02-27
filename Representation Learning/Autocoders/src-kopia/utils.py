from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from tqdm.auto import tqdm

from .ae import Autoencoder
from .dataset import SampledMNISTData


def extract_representations(
    model: Autoencoder,
    dataset: SampledMNISTData,
) -> dict:
    output = {
        "train": {"z": [], "y": []},
        "test": {"z": [], "y": []},
    }

    for split, loader in (
        ("train", dataset.train_dataloader()),
        ("test", dataset.test_dataloader()),
    ):
        for x, y in tqdm(loader):
            with torch.no_grad():
                output[split]["z"].append(model(x))
                output[split]["y"].append(y)

        output[split]["z"] = torch.cat(output[split]["z"], dim=0)
        output[split]["y"] = torch.cat(output[split]["y"], dim=0)

    return output


def evaluate_linear(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor
) -> Tuple[float, float]:
    lr = LogisticRegression(max_iter=500)
    lr.fit(z_train, y_train)

    auc_train = roc_auc_score(
        y_true=y_train,
        y_score=lr.predict_proba(z_train),
        multi_class="ovr",
    )
    auc_test = roc_auc_score(
        y_true=y_test,
        y_score=lr.predict_proba(z_test),
        multi_class="ovr",
    )

    return auc_train, auc_test


def visualize_latent_spaces(representations: dict):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

    for split, ax in zip(("train", "test"), axs.ravel()):
        z = representations[split]["z"]
        label = representations[split]["y"]
        sns.scatterplot(
            x=z[:, 0],
            y=z[:, 1],
            hue=label,
            ax=ax,
            legend="full",
            palette="Spectral",
            size=5,
        )
        ax.set(title=split.title())


def visualize_random_sample(
    model: Autoencoder,
    dataset,
    num_samples: int = 10,
    seed: int = 42,
):
    np.random.seed(seed)
    indices = np.random.choice(range(len(dataset)), size=num_samples)

    fig, axs = plt.subplots(nrows=2, ncols=num_samples, figsize=(15, 5))

    for i, sample_idx in enumerate(indices):
        org_img, label = dataset[sample_idx]
        axs[0, i].imshow(org_img[0], cmap="gray")
        axs[0, i].set(title=f"Digit: {label}", xticks=[], yticks=[])

        with torch.no_grad():
            rec_img = model.reconstruct(org_img.unsqueeze(dim=0))
        axs[1, i].imshow(rec_img[0][0], cmap="gray")
        axs[1, i].set(xticks=[], yticks=[])

        if i == 0:
            axs[0, i].set(ylabel="Original")
            axs[1, i].set(ylabel="Reconstruction")
