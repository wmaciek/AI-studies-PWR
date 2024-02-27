from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class BaseAE(pl.LightningModule):

    def __init__(self, hparams, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.encoder = encoder
        self.decoder = decoder

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx: int):
        out = {"loss": self._common_step(batch)}
        self.training_step_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        avg_loss = self._summarize_outputs(self.training_step_outputs)
        self.training_step_outputs.clear()

        self.log("step", self.trainer.current_epoch)
        self.log("train/loss", avg_loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx: int):
        out = {"loss": self._common_step(batch)}
        self.validation_step_outputs.append(out)

    def on_validation_epoch_end(self):
        avg_loss = self._summarize_outputs(self.validation_step_outputs)
        self.validation_step_outputs.clear()

        self.log("step", self.trainer.current_epoch)
        self.log("val/loss", avg_loss, on_epoch=True, on_step=False)

    @abstractmethod
    def _common_step(self, batch) -> torch.Tensor:
        pass

    @staticmethod
    def _summarize_outputs(outputs):
        losses = [out["loss"].detach().cpu().numpy() for out in outputs]

        avg_loss = np.mean(losses)

        return avg_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
