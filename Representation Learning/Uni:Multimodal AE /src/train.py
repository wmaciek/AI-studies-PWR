from typing import Type

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import DataModule


def train_model(
    model_cls: Type[pl.LightningModule],
    hparams,
    datamodule: DataModule,
):
    pl.seed_everything(42)

    model = model_cls(hparams)

    model_chkpt = ModelCheckpoint(
        dirpath=f"./data/checkpoints/{hparams['name']}/",
        filename="model",
        monitor="val/loss",
        mode="min",
        verbose=False,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            save_dir="./data/logs",
            name=hparams["name"],
            default_hp_metric=False,
        ),
        callbacks=[model_chkpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        max_epochs=hparams["num_epochs"],
    )

    trainer.fit(model=model, datamodule=datamodule)


@torch.no_grad()
def extract_embeddings(
    model_cls: Type[pl.LightningModule],
    name: str,
    datamodule: DataModule,
):
    best_model = model_cls.load_from_checkpoint(
        checkpoint_path=f"./data/checkpoints/{name}/model.ckpt"
    )
    best_model.eval()

    z = []

    for batch in datamodule.all_dataloader():
        z.append(best_model(batch))

    return torch.cat(z, dim=0)
