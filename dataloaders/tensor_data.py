import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from lightning import LightningDataModule
from torchvision.datasets import MNIST
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class TensorDataModule(LightningDataModule):
    def __init__(
        self,
        train_filepath: str,
        val_filepath: str,
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data: TensorDataPipe | None = None
        self.val_data: TensorDataPipe | None = None

    def setup(self, stage: str) -> None:
        self.train_data = TensorDataPipe(self.hparams.train_filepath)
        self.val_data = TensorDataPipe(self.hparams.val_filepath)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )


class TensorDataPipe(MapDataPipe):
    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.data = torch.load(filepath, map_location="cpu", mmap=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> FloatTensor:
        return self.data[index]
