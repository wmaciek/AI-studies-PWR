from torch import nn

from .ae import Autoencoder


class MLPEncoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )

    def forward(self, img):
        return self._layers(img)


class MLPDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()

        self._layers = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1 * 28 * 28),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self._layers(z)


class MLPAutoencoder(Autoencoder):

    def __init__(self, latent_dim: int):
        super().__init__(
            encoder=MLPEncoder(latent_dim=latent_dim),
            decoder=MLPDecoder(latent_dim=latent_dim),
        )
