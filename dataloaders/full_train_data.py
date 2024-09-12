from typing import Literal, get_args
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`IterableDataset` has `__len__` defined.*")

class FullDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 0,
        val_ratio: float = 0.2,
        scramble_data_by: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.data = FullDataPipe(
            data_dir=self.hparams.data_dir,
            val_ratio=self.hparams.val_ratio,
            scramble_data_by=self.hparams.scramble_data_by,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class FullDataPipe(MapDataPipe):
    Mode = Literal["train", "val"]

    def __init__(
        self,
        data_dir: str,
        val_ratio: float = 0.2,
        scramble_data_by: str | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio

        self.data = {
            f.replace(".pt", ""): torch.load(
                os.path.join(data_dir, f), map_location="cpu"
            )
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        }

        self.total_size = self.data[list(self.data.keys())[0]].shape[0]
        self.train_size = int(self.total_size * (1 - val_ratio))
        self.val_size = self.total_size - self.train_size

        self.mode = "train"

        if scramble_data_by is None:
            idx_scrambled = list(range(self.total_size))
            np.random.shuffle(idx_scrambled)
            self._idx_map = lambda idx: idx_scrambled[idx]
        else:
            self._idx_map = self.scramble_data(by=scramble_data_by)

    def scramble_data(self, by: str):
        data = self.data[by]
        assert data.dtype == torch.int64
        idx_scrambled = {}
        for i, val in enumerate(data):
            val = tuple(val.tolist())
            if val not in idx_scrambled:
                idx_scrambled[val] = []
            idx_scrambled[val].append(i)
        idx_scrambled = list(idx_scrambled.values())
        np.random.shuffle(idx_scrambled)  # Scrambled the variables
        for vals in idx_scrambled:  # Scramble within the same variable
            np.random.shuffle(vals)
        idx_scrambled = [val for vals in idx_scrambled for val in vals]
        return lambda idx: idx_scrambled[idx]

    def set_mode(self, mode: Mode):
        assert mode in get_args(FullDataPipe.Mode)
        self.mode = mode

    def __len__(self) -> int:
        if self.mode == "train":
            return self.train_size
        elif self.mode == "val":
            return self.val_size
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __getitem__(self, index) -> dict[str, Tensor]:
        if self.mode == "val":
            index += self.train_size
        index = self._idx_map(index)
        return {k: v[index] for k, v in self.data.items()}