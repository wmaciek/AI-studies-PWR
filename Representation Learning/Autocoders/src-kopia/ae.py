from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(LightningModule):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        return self._encoder(x)

    def reconstruct(self, x):
        return self._decoder(z=self(x))

    def training_step(self, batch, batch_size):
        x, _ = batch
        xr = self.reconstruct(x)

        loss = F.mse_loss(input=xr, target=x)
        self.log("train/mse", loss.detach().item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
