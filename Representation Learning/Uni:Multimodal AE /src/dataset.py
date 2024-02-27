import os
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MemeDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        return dict(self._df.iloc[index])


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        path: str = "data/processed.pkl",
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()

        df = pd.read_pickle(path)
        train_df, val_df = train_test_split(
            df,
            train_size=0.8,
            random_state=seed,
        )

        self.df = {"train": train_df, "val": val_df, "all": df}
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")

    def all_dataloader(self) -> DataLoader:
        return self._dataloader("all")

    def _dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            MemeDataset(self.df[split]),
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=int(os.environ.get("NUM_WORKERS", 0)),
        )
