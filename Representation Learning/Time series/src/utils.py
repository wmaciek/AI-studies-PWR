import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils import data
import torch
from sklearn.manifold import TSNE


def create_simulated_dataset(dataset, window_size, batch_size):
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]

    x_valid = dataset["x_val"]
    y_valid = dataset["y_val"]

    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    datasets = []
    for x, y, in [(x_train, y_train), (x_test, y_test), (x_valid, y_valid)]:
        T = x.shape[-1]
        windows = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    trainset, testset, validset = datasets[0], datasets[1], datasets[2]
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def visualize_embeddings(x_all, y_all, encoder, window_size):
    x_all = x_all.numpy()
    y_all = y_all.numpy()

    n_all = len(x_all)
    inds = np.random.randint(0, x_all.shape[-1] - window_size, n_all)
    windows = np.array([
        x_all[int(i % n_all), :, ind:ind + window_size]
        for i, ind in enumerate(inds)
    ])
    windows_state = [
        np.round(np.mean(y_all[i % n_all, ind:ind + window_size], axis=-1))
        for i, ind in enumerate(inds)
    ]

    with torch.no_grad():
        encodings = encoder(torch.tensor(windows))

    tnc_embedding = TSNE(n_components=2).fit_transform(encodings.cpu().numpy())
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    sns.scatterplot(
        x=original_embedding[:, 0],
        y=original_embedding[:, 1],
        hue=windows_state,
        ax=axs[0],
    )
    axs[0].set_title("(T-SNE) Wartość sygnału", fontweight="bold")

    sns.scatterplot(
        x=tnc_embedding[:, 0],
        y=tnc_embedding[:, 1],
        hue=windows_state,
        ax=axs[1],
    )
    axs[1].set_title("(T-SNE) Reprezentacje TNC", fontweight="bold")

    return fig
