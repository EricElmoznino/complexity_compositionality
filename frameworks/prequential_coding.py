from abc import ABC, abstractmethod
import os
import torch
from torch import nn, Tensor
from lightning import LightningModule
from datasets.prequential_data import PrequentialDataPipe, PrequentialDataModule


class PrequentialCoding(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        interval_patience: int = 15,
        lr: float = 1e-3,
        model_cache_dir: str | None = None,
        include_initial_length: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["predictor"])

        self.model = model

        self.interval_patience = interval_patience
        self.interval_epochs_since_improvement = 0
        self.interval_best_loss = torch.inf
        self.interval_errors = []

        self.model_cache_dir = model_cache_dir
        if model_cache_dir is None and "SLURM_TMPDIR" in os.environ:
            self.model_cache_dir = os.environ["SLURM_TMPDIR"]

    @abstractmethod
    def forward(self, data: dict[str, Tensor], sum: bool = False):
        # Returns the negative log-likelihood of the data given the model,
        # either summed across tensor dimensions or averaged.
        pass

    @property
    @abstractmethod
    def initial_length(self) -> float:
        # Returns the initial length of the unmodeled first data increment.
        pass

    def training_step(self, data, batch_idx):
        loss = self.forward(data)
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        loss = self.forward(data)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    def on_train_start(self):
        if self.hparams.include_initial_length:
            self.log("data encoded", self.dataset.data_encoded)
            self.log("K/K(interval i)", self.initial_length)

    def on_validation_epoch_start(self):
        self.dataset.set_mode("val")

    def on_validation_epoch_end(self):
        self.dataset.set_mode("train")

    def on_train_epoch_end(self):
        val_loss = self.trainer.callback_metrics["loss/val"]
        if val_loss < self.interval_best_loss:
            self.interval_best_loss = val_loss
            self.interval_epochs_since_improvement = 0
            if self.model_cache_dir is not None:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_cache_dir, "best.pt"),
                )
        else:
            self.interval_epochs_since_improvement += 1

        if self.interval_epochs_since_improvement >= self.interval_patience:
            if self.model_cache_dir is not None:
                self.model.load_state_dict(
                    torch.load(os.path.join(self.model_cache_dir, "best.pt"))
                )
            self.dataset.increment_data_size()
            self.compute_length(mode="encode")
            self.interval_epochs_since_improvement = 0
            self.interval_best_loss = torch.inf
            if not self.dataset.done:
                self.model.reset_params()

        if self.dataset.done:
            self.compute_length(mode="all_train")
            self.trainer.should_stop = True

    def compute_length(self, mode: str):
        prev_mode = self.dataset.mode
        self.dataset.set_mode(mode)
        dataloader = self.trainer.datamodule.val_dataloader()

        neg_logp = 0
        for data in dataloader:
            neg_logp += self.forward(data, sum=True).detach().cpu().item()

        if mode == "encode":
            self.interval_errors.append(neg_logp)
            self.log("data encoded", self.dataset.data_encoded)
            self.log("K/K(interval i)", neg_logp)
        else:
            k_data = sum(self.interval_errors)
            self.log("K/K(Data)", k_data)
            self.log("K/K(Data|f)", neg_logp)
            self.log("K/K(f)", k_data - neg_logp)

        self.dataset.set_mode(prev_mode)

    @property
    def dataset(self) -> PrequentialDataPipe:
        dm: PrequentialDataModule = self.trainer.datamodule
        return dm.data

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
