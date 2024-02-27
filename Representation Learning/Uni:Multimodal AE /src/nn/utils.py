from typing import List, Type

from torch import nn


def get_unimodal_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    last_activation: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dims[0]),
        nn.ReLU(inplace=True),
        *[
            layer
            for idx in range(len(hidden_dims) - 1)
            for layer in (nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]), nn.ReLU(inplace=True))
        ],
        nn.Linear(hidden_dims[-1], out_dim),
        last_activation(),
    )

