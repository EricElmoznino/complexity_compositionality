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
        sentence_length: int,
        vocab_size: int,
        z_dim: int,
        disentanglement: int,
        discrete: bool,
        num_attributes: int | None,
        num_vals: int | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if (
            not os.path.exists(os.path.join(self.hparams.data_dir, "w.pt"))
            or not os.path.exists(os.path.join(self.hparams.data_dir, "z.pt")),
        ):
            self.generate_data()

    def generate_data(self):
        num_positions = self.hparams.sentence_length // self.hparams.disentanglement
        num_entries = self.hparams.vocab_size**self.hparams.disentanglement
        dim = self.hparams.z_dim // num_positions
        lookup_table = torch.randn(num_positions, num_entries, dim)

        w = torch.combinations(
            torch.LongTensor(range(self.hparams.vocab_size)),
            r=self.hparams.sentence_length,
            with_replacement=True,
        )
        w = w[torch.randperm(w.size(0))]

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
