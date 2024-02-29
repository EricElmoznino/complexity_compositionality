import os
import torch
from torch import FloatTensor
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from torchvision.datasets import MNIST
from torchvision import transforms
import warnings
from datasets.prequential_data import PrequentialDataModule

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class DebugPrequentialDataModule(PrequentialDataModule):
    def __init__(
        self,
        num_words: int,
        vocab_size: int,
        disentanglement: int,
        discrete: bool,
        z_dim: int | None,
        num_attributes: int | None,
        num_vals: int | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if not os.path.exists(self.hparams.data_dir):
            os.mkdir(self.hparams.data_dir)
            self.generate_data()

    def generate_data(self):
        num_positions = self.hparams.num_words // self.hparams.disentanglement
        num_entries = self.hparams.vocab_size**self.hparams.disentanglement

        if self.hparams.discrete:
            dim = self.hparams.num_attributes // num_positions
            lookup_table = torch.randint(
                0, self.hparams.num_vals, (num_positions, num_entries, dim)
            )
        else:
            dim = self.hparams.z_dim // num_positions
            lookup_table = torch.randn(num_positions, num_entries, dim)

        w = torch.cartesian_prod(
            *[
                torch.LongTensor(range(self.hparams.vocab_size))
                for _ in range(self.hparams.num_words)
            ]
        )
        w = w[torch.randperm(w.shape[0])]

        z = []
        for wi in w:
            zi = []
            wi = wi.split(self.hparams.disentanglement)
            for pos, wi_pos in enumerate(wi):
                entry = (
                    (
                        wi_pos
                        * (
                            self.hparams.vocab_size
                            ** torch.arange(self.hparams.disentanglement)
                        )
                    )
                    .sum()
                    .int()
                )
                zi.append(lookup_table[pos, entry])
            z.append(torch.cat(zi))
        z = torch.stack(z)

        torch.save(w, os.path.join(self.hparams.data_dir, "w.pt"))
        torch.save(z, os.path.join(self.hparams.data_dir, "z.pt"))


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data: MNIST | None = None
        self.val_data: MNIST | None = None

    def setup(self, stage: str) -> None:
        transf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        class MNISTWrapper(MNIST):
            def __getitem__(self, index) -> FloatTensor:
                img, _ = super().__getitem__(index)
                return img

        self.train_data = MNISTWrapper(
            root=self.hparams.data_path,
            train=True,
            download=True,
            transform=transf,
        )
        self.val_data = MNISTWrapper(
            root=self.hparams.data_path,
            train=False,
            download=True,
            transform=transf,
        )

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
