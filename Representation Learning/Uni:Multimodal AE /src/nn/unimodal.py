import torch
from torch import nn
from torch.nn import functional as F

from src.nn.ae import BaseAE
from src.nn.utils import get_unimodal_mlp


class UnimodalAE(BaseAE):

    def __init__(self, hparams):
        super().__init__(
            hparams=hparams,
            encoder=get_unimodal_mlp(
                in_dim=hparams["data_dim"],
                hidden_dims=hparams["hidden_dims"],
                out_dim=hparams["emb_dim"],
                last_activation=nn.Tanh,
            ),
            decoder=get_unimodal_mlp(
                in_dim=hparams["emb_dim"],
                hidden_dims=hparams["hidden_dims"][::-1],
                out_dim=hparams["data_dim"],
                last_activation=nn.Identity,
            ),
        )

    def forward(self, batch) -> torch.Tensor:
        return self.encoder(batch[self.hparams["modality_name"]])

    def _common_step(self, batch) -> torch.Tensor:
        x = batch[self.hparams["modality_name"]]

        z = self.encoder(x)
        x_rec = self.decoder(z)

        loss = F.mse_loss(input=x_rec, target=x)

        return loss

