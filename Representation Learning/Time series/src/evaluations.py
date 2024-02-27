import torch

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.utils import create_simulated_dataset


def evaluate_model(encoder: torch.nn.Module, dataset, window_size: int):
    encoder.eval()

    train_loader, valid_loader, test_loader = create_simulated_dataset(
        dataset=dataset, window_size=window_size, batch_size=100
    )

    z_train, y_train = _extract_embeddings(encoder, train_loader)
    z_val, y_val = _extract_embeddings(encoder, valid_loader)
    z_test, y_test = _extract_embeddings(encoder, test_loader)

    aucs = _calculate_classification_auc(
        z_train=z_train, y_train=y_train,
        z_val=z_val, y_val=y_val,
        z_test=z_test, y_test=y_test,
    )

    # Clustering
    z_all = torch.cat([z_train, z_val, z_test])

    clustering_metrics = _calculate_clustering_metrics(z_all)

    return {"auc": aucs, "cluster": clustering_metrics}


@torch.no_grad()
def _extract_embeddings(encoder, loader):
    embs, ys = [], []

    for x, y in loader:
        emb = encoder(x)
        embs.append(emb)
        ys.append(y)

    return torch.cat(embs), torch.cat(ys)


def _calculate_classification_auc(z_train, y_train, z_val, y_val, z_test, y_test):
    _lr = LogisticRegression()
    _lr.fit(z_train, y_train)

    def _auc(z, y):
        return roc_auc_score(
            y_true=y,
            y_score=_lr.predict_proba(z),
            multi_class="ovr",
        )

    aucs = {
        "train": _auc(z=z_train, y=y_train),
        "val": _auc(z=z_val, y=y_val),
        "test": _auc(z=z_test, y=y_test),
    }

    return aucs


def _calculate_clustering_metrics(z_all):
    kmeans = KMeans(n_clusters=4, random_state=1).fit(z_all)
    cluster_labels = kmeans.labels_

    clustering_metrics = {
        "silhouette": silhouette_score(z_all, cluster_labels),
        "davies_bouldin": davies_bouldin_score(z_all, cluster_labels),
    }

    return clustering_metrics
