import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def get_default_trainer(
    num_epochs: int,
    model_name: str,
    quiet: bool = False,
) -> pl.Trainer:
    if quiet:
        kwargs = dict(
            logger=False,
            log_every_n_steps=-1,
            check_val_every_n_epoch=-1,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    else:
        kwargs = dict(
            logger=TensorBoardLogger(
                save_dir="./data/logs/",
                name=model_name,
                default_hp_metric=False,
            ),
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
        )

    return pl.Trainer(
        enable_checkpointing=False,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        **kwargs,
    )
