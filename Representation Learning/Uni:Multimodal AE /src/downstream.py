from typing import List, Tuple, Type

import pandas as pd
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .dataset import DataModule
from .train import extract_embeddings


DOWNSTREAM_TASKS = (
    "humour",
    "sarcasm",
    "offensive",
    "motivational",
    "sentiment",
)


def evaluate_classification(
    model_names: List[Tuple[Type[pl.LightningModule], str]],
    datamodule: DataModule,
    seed: int = 42,
) -> pd.DataFrame:
    out = []
    for model_cls, name in model_names:
        z = extract_embeddings(
            model_cls=model_cls,
            name=name,
            datamodule=datamodule,
        )
        for task in DOWNSTREAM_TASKS:
            labels = datamodule.df["all"][f"label_{task}"]

            z_train, z_test, y_train, y_test = train_test_split(
                z, labels,
                train_size=0.6,
                random_state=seed,
            )

            lr = LogisticRegression()
            lr.fit(z_train, y_train)

            auc_train = auc(y_score=lr.predict_proba(z_train), y_true=y_train)
            auc_test = auc(y_score=lr.predict_proba(z_test), y_true=y_test)

            out.append({
                "model": name,
                "task": task,
                "train_AUC": f"{auc_train * 100.:.2f} [%]",
                "test_AUC": f"{auc_test * 100.:.2f} [%]",
            })

    return (
        pd.DataFrame.from_records(out)
        .pivot(index="model", columns="task")
        .reindex([name for _, name in model_names])
    )


def auc(y_score, y_true):
    if len(y_true.unique()) == 2:  # Binary case
        y_score = y_score[:, 1]

    return roc_auc_score(
        y_score=y_score,
        y_true=y_true,
        multi_class="ovr",
    )
