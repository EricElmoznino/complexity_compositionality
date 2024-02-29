from abc import ABC, abstractmethod
import os
import torch
from torch import nn, Tensor, FloatTensor
from torch.nn import functional as F
from lightning import LightningModule
from datasets.prequential_data import PrequentialDataPipe, PrequentialDataModule
from models.decoders import SentenceDecoder


class PrequentialCoding(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        interval_patience: int = 15,
        interval_patience_tol: float = 0.0,
        lr: float = 1e-3,
        model_cache_dir: str | None = None,
        include_initial_length: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.interval_patience = interval_patience
        self.interval_patience_tol = interval_patience_tol
        self.interval_epochs_since_improvement = 0
        self.interval_best_loss = torch.inf
        self.interval_errors = []

        self.model_cache_dir = model_cache_dir
        if model_cache_dir is None and "SLURM_TMPDIR" in os.environ:
            self.model_cache_dir = os.environ["SLURM_TMPDIR"]

    @abstractmethod
    def forward(self, data: dict[str, Tensor], sum: bool = False) -> FloatTensor:
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
        # Check for improvement on validation set
        val_loss = self.trainer.callback_metrics["loss/val"]
        if val_loss < self.interval_best_loss - self.interval_patience_tol:
            self.interval_best_loss = val_loss
            self.interval_epochs_since_improvement = 0
            if self.model_cache_dir is not None:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_cache_dir, "best.pt"),
                )
        else:
            self.interval_epochs_since_improvement += 1

        # If no improvement, stop training on this current set of data
        if self.interval_epochs_since_improvement >= self.interval_patience:
            if self.model_cache_dir is not None:
                self.model.load_state_dict(
                    torch.load(os.path.join(self.model_cache_dir, "best.pt"))
                )

            # Encode the next increment of data
            if not self.dataset.done:
                self.dataset.increment_data_size()
                self.compute_length(mode="encode")
                self.interval_epochs_since_improvement = 0
                self.interval_best_loss = torch.inf
                self.reset_model_params()
            # Get final loss across the whole dataset
            else:
                self.compute_length(mode="all_train")
                self.trainer.should_stop = True

    def compute_length(self, mode: str):
        prev_mode = self.dataset.mode
        self.dataset.set_mode(mode)
        dataloader = self.trainer.datamodule.val_dataloader()

        neg_logp = 0
        for data in dataloader:
            data = {k: v.to(self.device) for k, v in data.items()}
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

    def reset_model_params(self):
        def reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.model.apply(reset_module)

    @property
    def dataset(self) -> PrequentialDataPipe:
        dm: PrequentialDataModule = self.trainer.datamodule
        return dm.data

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


##############################################################
######################## Subclasses ##########################
##############################################################


class PrequentialCodingSentenceDecoder(PrequentialCoding):
    def __init__(
        self,
        *args,
        discrete_z: bool = False,
        z_num_attributes: int | None = None,
        z_num_vals: int | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.model: SentenceDecoder  # Just for type annotation

    def forward(self, data: dict[str, Tensor], sum: bool = False) -> FloatTensor:
        w, z_true = data["w"], data["z"]
        z_mu, z_logstd = self.model(w)
        if self.hparams.discrete_z:
            z_logits = z_mu.view(
                -1, self.hparams.z_num_attributes, self.hparams.z_num_vals
            )
            dist = torch.distributions.Categorical(logits=z_logits)
        else:
            dist = torch.distributions.Normal(z_mu, z_logstd.exp())
        logp = dist.log_prob(z_true)
        logp = logp.sum() if sum else logp.mean()
        return -logp

    @property
    def initial_length(self) -> float:
        z_initial = self.dataset.data["z"][: self.dataset.data_sizes[0]]
        if self.hparams.discrete_z:
            z_marginal = (
                F.one_hot(z_initial, num_classes=self.hparams.z_num_vals)
                .float()
                .mean(dim=0)
            )
            z_marginal = torch.distributions.Categorical(probs=z_marginal)
        else:
            z_marginal_mu, z_marginal_std = z_initial.mean(dim=0), z_initial.std(dim=0)
            z_marginal = torch.distributions.Normal(z_marginal_mu, z_marginal_std)
        logp = z_marginal.log_prob(z_initial)
        return -logp.sum().item()
