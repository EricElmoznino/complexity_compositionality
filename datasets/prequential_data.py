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


class PrequentialDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        min_data_size: int,
        val_size: int,
        data_increment: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        self.data = PrequentialDataPipe(
            data_dir=self.hparams.data_dir,
            min_data_size=self.hparams.min_data_size,
            val_size=self.hparams.val_size,
            data_increment=self.hparams.data_increment,
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


class PrequentialDataPipe(MapDataPipe):
    Mode = Literal["train", "val", "encode", "all_train"]

    def __init__(
        self,
        data_dir: str,
        min_data_size: int,
        val_size: int,
        data_increment: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.min_data_size = min_data_size
        self.val_size = val_size
        self.data_increment = data_increment
        self.data = {
            f.replace(".pt", ""): torch.load(
                os.path.join(data_dir, f), map_location="cpu", mmap=True
            )
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        }
        self.train_size = self.data[list(self.data.keys())[0]].shape[0] - val_size
        self.data_sizes = np.arange(min_data_size, self.train_size + 1, data_increment)
        self._data_size_idx = 0
        self.mode = "train"

    def set_mode(self, mode: Mode):
        assert mode in get_args(PrequentialDataPipe.Mode)
        if mode == "train" or mode == "val":
            assert not self.done
        elif mode == "encode":
            assert self._data_size_idx > 0
        elif mode == "all_train":
            assert self.done
        self.mode = mode

    @property
    def done(self) -> bool:
        return self._data_size_idx >= len(self.data_sizes) - 1

    @property
    def data_encoded(self) -> float:
        return float(self.data_sizes[self._data_size_idx])

    def increment_data_size(self):
        assert not self.done
        self._data_size_idx += 1

    def __len__(self) -> int:
        # Only data currently being trained on
        if self.mode == "train":
            return self.data_sizes[self._data_size_idx]
        # All data not currently trained on, used to decide when to move to the next data interval
        elif self.mode == "val":
            return self.val_size
        # Only next data interval, used for prequential code length
        elif self.mode == "encode":
            return (
                self.data_sizes[self._data_size_idx]
                - self.data_sizes[self._data_size_idx - 1]
            )
        elif self.mode == "all_train":
            return self.train_size
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __getitem__(self, index) -> dict[str, Tensor]:
        if self.mode == "val":
            index += self.train_size
        elif self.mode == "encode":
            index += self.data_sizes[self._data_size_idx - 1]
        return {k: v[index] for k, v in self.data.items()}
