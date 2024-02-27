from typing import List, Tuple

import torch
from torch_geometric.nn.models import Node2Vec
from tqdm.auto import tqdm


def train_random_walk_model(
    edge_index: torch.Tensor,
    p: float,
    q: float,

    # Default training params
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    num_workers: int = 4,

    # Default method params
    embedding_dim: int = 128,
    walk_length: int = 20,
    context_size: int = 5,
    walks_per_node: int = 1,
    num_negative_samples: int = 1,

    # Progress bars
    quiet: bool = False,
) -> Tuple[Node2Vec, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_negative_samples=num_negative_samples,
    ).to(device)

    if not quiet:
        num_parameters = list(model.parameters())[0].numel()
        print(f"Liczba parametrów modelu: {num_parameters:,}")

    # Training
    loader = model.loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for _ in tqdm(iterable=range(num_epochs), desc="Epochs", disable=quiet):
        total_loss = 0

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss / len(loader))

    return model, losses


def get_representations(model: Node2Vec) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        z = model().cpu().detach()

    return z
