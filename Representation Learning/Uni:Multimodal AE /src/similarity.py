from typing import Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def get_most_similar(
    x: torch.Tensor,
    anchor: torch.Tensor,
    metric: str = "cosine",
    num_neighbors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:

    nn = NearestNeighbors(
        n_neighbors=num_neighbors + 1,
        metric=metric
    ).fit(x)

    similarities, indices = nn.kneighbors(anchor.reshape(1, -1))
    return similarities[0][1:], indices[0][1:]
