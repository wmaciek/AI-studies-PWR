from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms as T


class SampledMNISTData(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "./data/mnist",
        batch_size: int = 64,
        num_samples_per_class: int = -1,
        seed: int = 42,
    ):
        super().__init__()
        self.dims = (1, 28, 28)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.seed = seed

        self.mnist_train = None
        self.mnist_test = None
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        train = MNIST(self.data_dir, train=True, transform=self.transform)
        test = MNIST(self.data_dir, train=False, transform=self.transform)

        if self.num_samples_per_class == -1:
            self.mnist_train = train
            self.mnist_test = test
        else:
            self.mnist_train = subsample(
                mnist=train,
                num_samples_per_class=self.num_samples_per_class,
                seed=self.seed,
            )
            self.mnist_test = subsample(
                mnist=test,
                num_samples_per_class=self.num_samples_per_class,
                seed=self.seed,
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
        )


def subsample(
    mnist: MNIST,
    num_samples_per_class: int,
    seed: int,
) -> TensorDataset:
    np.random.seed(seed)
    unique_labels = sorted(mnist.targets.unique())

    sampled_data = []
    sampled_labels = []

    for i in unique_labels:
        label_idxs = (mnist.targets == i).int().nonzero().squeeze().tolist()
        sampled_idxs = np.random.choice(label_idxs, size=num_samples_per_class)

        for idx in sampled_idxs:
            data, label = mnist[idx]
            sampled_data.append(data)
            sampled_labels.append(label)

    sampled_data = torch.stack(sampled_data, dim=0)
    sampled_labels = torch.tensor(sampled_labels)

    return TensorDataset(sampled_data, sampled_labels)
