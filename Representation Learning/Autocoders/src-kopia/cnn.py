import torch
from torch import nn

from .ae import Autoencoder

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


class CNNEncoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()

        self._layers = nn.Sequential(
            # Convolution
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            # Invariant pooling
            nn.MaxPool2d(kernel_size=3),
            # Flattening
            nn.Flatten(),
            # MLP
            nn.Linear(32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, img):
        return self._layers(img)


class CNNDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()

        self._layers = nn.Sequential(
            # MLP
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            # Reshape into image
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            # Deconvolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self._layers(z)


class CNNAutoencoder(Autoencoder):

    def __init__(self, latent_dim: int):
        super().__init__(
            encoder=CNNEncoder(latent_dim=latent_dim),
            decoder=CNNDecoder(latent_dim=latent_dim),
        )
