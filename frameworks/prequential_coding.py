from abc import ABC, abstractmethod
from typing import Any
import math
import os
from copy import deepcopy
import torch
from torch import nn, Tensor, FloatTensor, LongTensor
from torch.nn import functional as F
from lightning import LightningModule
from torchmetrics.classification import MulticlassExactMatch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from dataloaders.prequential_data import PrequentialDataPipe, PrequentialDataModule
from models.decoders import SentenceDecoder
from utils import skellam


class PrequentialCoding(ABC, LightningModule):
    def __init__(
        self,
        model: nn.Module,
        interval_patience: int = 15,
        interval_patience_tol: float = 0.0,
        lr: float = 1e-3,
        model_cache: str | None = None,
        include_initial_length: bool = True,
        allow_final_overfit: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.interval_patience = interval_patience
        self.interval_patience_tol = interval_patience_tol
        self.interval_epochs_since_improvement = 0
        self.interval_best_loss = torch.inf
        self.interval_errors = []

        self.model_cache = model_cache
        if model_cache is None:
            self.best_model = None
        elif model_cache == "slurm" and "SLURM_TMPDIR" in os.environ:
            self.model_cache = os.environ["SLURM_TMPDIR"]
        else:
            assert os.path.exists(model_cache)

    @abstractmethod
    def forward(self, data: dict[str, Tensor]) -> Any:
        # Model predictions
        pass

    @abstractmethod
    def nll(
        self,
        pred: Any,
        data: dict[str, Tensor],
        encode: bool = False,
    ) -> FloatTensor:
        # Returns the negative log-likelihood of the data given the model's predictions,
        # either for the loss or for encoding.
        pass

    def log_additional_performance_metrics(
        self,
        pred: Any,
        data: dict[str, Tensor],
        stage: str,
    ) -> None:
        return

    @abstractmethod
    def compute_naive_length(self) -> float:
        # Returns the length of data using naive statistics.
        pass

    def training_step(self, data, batch_idx):
        pred = self.forward(data)
        loss = self.nll(pred, data)
        self.log(
            "training/train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_additional_performance_metrics(pred, data, "train")
        return loss

    def validation_step(self, data, batch_idx):
        pred = self.forward(data)
        loss = self.nll(pred, data)
        self.log("training/val_loss", loss, prog_bar=True)
        self.log_additional_performance_metrics(pred, data, "val")
        return loss

    def on_train_start(self):
        if self.hparams.include_initial_length:
            initial_length = self.compute_naive_length()
            self.interval_errors.append(initial_length)
            self.log("data encoded", self.dataset.data_encoded)
            self.log("K/K(interval i)", initial_length)

    def on_validation_epoch_start(self):
        self.dataset.set_mode("val")

    def on_validation_epoch_end(self):
        self.dataset.set_mode("train")

    def on_train_epoch_end(self):
        # Pick either validation set or training set for stopping criteria
        if self.dataset.done and self.hparams.allow_final_overfit:
            loss = self.trainer.callback_metrics["training/train_loss"]
        else:
            loss = self.trainer.callback_metrics["training/val_loss"]

        # Check for improvement
        if loss < self.interval_best_loss - self.interval_patience_tol:
            self.interval_best_loss = loss
            self.interval_epochs_since_improvement = 0
            if self.model_cache is None:
                self.best_model = deepcopy(self.model.state_dict())
            else:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_cache, "best.pt"),
                )
        else:
            self.interval_epochs_since_improvement += 1

        # If no improvement, stop training on this current set of data
        if self.interval_epochs_since_improvement >= self.interval_patience:
            if self.model_cache is None:
                self.model.load_state_dict(self.best_model)
            else:
                self.model.load_state_dict(
                    torch.load(os.path.join(self.model_cache, "best.pt"))
                )
            # Encode the next increment of data
            if not self.dataset.done:
                self.dataset.increment_data_size()
                self.compute_length(mode="encode")
                self.interval_epochs_since_improvement = 0
                self.interval_best_loss = torch.inf
                self.reset_model_params()
                self.trainer.optimizers = [self.configure_optimizers()]
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
            pred = self.forward(data)
            neg_logp += self.nll(pred, data, encode=True).detach().cpu().item()

        if mode == "encode":
            neg_logp = min(neg_logp, self.compute_naive_length())
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.model: SentenceDecoder  # Just for type annotation
        if discrete_z:
            self.train_accuracy = MulticlassExactMatch(num_classes=z_num_vals)
            self.val_accuracy = MulticlassExactMatch(num_classes=z_num_vals)

    def forward(
        self, data: dict[str, Tensor]
    ) -> FloatTensor | tuple[FloatTensor, FloatTensor]:
        w: LongTensor = data["w"]
        z_mu, z_logstd = self.model.forward(w)
        if self.hparams.discrete_z:
            z_logits = z_mu.view(
                -1, self.hparams.z_num_attributes, self.hparams.z_num_vals
            )
            return z_logits
        else:
            return z_mu, z_logstd

    def nll(
        self,
        pred: FloatTensor | tuple[FloatTensor, FloatTensor],
        data: dict[str, Tensor],
        encode: bool = False,
    ) -> FloatTensor:
        z_true = data["z"]
        if self.hparams.discrete_z:
            z_logits = pred
            dist = torch.distributions.Categorical(logits=z_logits)
            logp = dist.log_prob(z_true)
        else:
            z_mu, z_logstd = pred
            if encode:
                logp = skellam.approx_gaussian_logpmf(z_true, z_mu, z_logstd.exp())
            else:
                logp = torch.distributions.Normal(z_mu, z_logstd.exp()).log_prob(z_true)
        logp = logp.sum() if encode else logp.mean()
        return -logp

    def log_additional_performance_metrics(
        self,
        pred: FloatTensor | tuple[FloatTensor, FloatTensor],
        data: dict[str, Tensor],
        stage: str,
    ) -> None:
        if not self.hparams.discrete_z:
            return
        accuracy = self.train_accuracy if stage == "train" else self.val_accuracy
        z_logits = pred.transpose(1, 2)
        accuracy(z_logits, data["z"])
        self.log(
            f"training/{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def compute_naive_length(self) -> float:
        if self.dataset.data_size_idx == 0:
            z = self.dataset.data["z"][: self.dataset.data_sizes[0]]
        else:
            z = self.dataset.data["z"][
                self.dataset.data_sizes[
                    self.dataset.data_size_idx - 1
                ] : self.dataset.data_sizes[self.dataset.data_size_idx]
            ]
        if self.hparams.discrete_z:
            z_uniform = torch.distributions.Categorical(
                logits=torch.ones(
                    self.hparams.z_num_attributes, self.hparams.z_num_vals
                ).to(z.device)
            )
            logp = z_uniform.log_prob(z)
        else:
            z_marginal_mu, z_marginal_std = (
                z.mean(dim=0, keepdim=True).expand_as(z),
                z.std(dim=0, keepdim=True).expand_as(z),
            )
            logp = skellam.approx_gaussian_logpmf(z, z_marginal_mu, z_marginal_std)
        return -logp.sum().item()


class PrequentialCodingHuggingFaceSentence(PrequentialCoding):
    def __init__(
        self,
        *args,
        model_name: str,
        learn_embeddings: bool = False,
        short_vocab_size: int | None = None,
        **kwargs,
    ):
        model = SentenceTransformer(model_name)
        self.config = model[0].auto_model.config

        if learn_embeddings:
            assert short_vocab_size is not None
            self.config.vocab_size = short_vocab_size
            model[0].auto_model = AutoModel.from_config(self.config)
        else:
            model[0].auto_model.embeddings.requires_grad_(False)

        super().__init__(*args, model=model, **kwargs)
        self.save_hyperparameters(ignore=["model"])

        self.reset_model_params()

        self.z_marginal_mu: FloatTensor | None = None
        self.z_marginal_std: FloatTensor | None = None

    def on_train_start(self):
        super().on_train_start()
        z = self.dataset.data["z"][
            : self.dataset.data_sizes[0]
        ]  # First transmitted chunk's standard deviation used for Skellam
        self.z_marginal_mu = z.mean(dim=0, keepdim=True)
        self.z_marginal_std = z.std(dim=0, keepdim=True)

    def reset_model_params(self):
        init_state = AutoModel.from_config(self.config).state_dict()
        if not self.hparams.learn_embeddings:
            init_state = {k: v for k, v in init_state.items() if "embeddings" not in k}
        incompatible = self.model[0].auto_model.load_state_dict(
            init_state, strict=False
        )
        assert len(incompatible.unexpected_keys) == 0
        if self.hparams.learn_embeddings:
            assert len(incompatible.missing_keys) == 0
        else:
            assert all(["embeddings" in k for k in incompatible.missing_keys])

    def forward(self, data: dict[str, Tensor]) -> FloatTensor:
        if self.hparams.learn_embeddings:
            w: LongTensor = data["w_short"]
        else:
            w: LongTensor = data["w"]
        attention_mask = torch.ones_like(w)
        attention_mask[w == 1] = 0  # Assumes 1 is the padding token
        w = {"input_ids": w, "attention_mask": attention_mask}
        z_mu = self.model.forward(w)["sentence_embedding"]
        return z_mu

    def nll(
        self,
        pred: FloatTensor,
        data: dict[str, Tensor],
        encode: bool = False,
    ) -> FloatTensor:
        z_true, z_mu = data["z"], pred
        if encode:
            neg_logp = -skellam.approx_gaussian_logpmf(
                z_true, z_mu, self.z_marginal_std.expand_as(z_mu)
            )
            naive_neg_logp = -skellam.approx_gaussian_logpmf(
                z_true,
                self.z_marginal_mu.expand_as(z_true).to(z_true.device),
                self.z_marginal_std.expand_as(z_true).to(z_true.device),
            )
            neg_logp = torch.where(torch.isinf(neg_logp), naive_neg_logp, neg_logp)
            return neg_logp.sum()
        else:
            return F.mse_loss(z_mu, z_true)

    def compute_naive_length(self) -> float:
        if self.dataset.data_size_idx == 0:
            z = self.dataset.data["z"][: self.dataset.data_sizes[0]]
        else:
            z = self.dataset.data["z"][
                self.dataset.data_sizes[
                    self.dataset.data_size_idx - 1
                ] : self.dataset.data_sizes[self.dataset.data_size_idx]
            ]
        z_marginal_mu, z_marginal_std = (
            z.mean(dim=0, keepdim=True).expand_as(z),
            z.std(dim=0, keepdim=True).expand_as(z),
        )
        logp = skellam.approx_gaussian_logpmf(z, z_marginal_mu, z_marginal_std)
        return -logp.sum().item()
