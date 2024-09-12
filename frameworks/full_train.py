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


class FullTraining(ABC, LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        reset_model_params: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    @abstractmethod
    def forward(self, data: dict[str, Tensor]) -> Any:
        pass

    @abstractmethod
    def nll(
        self,
        pred: Any,
        data: dict[str, Tensor],
    ) -> FloatTensor:
        pass

    def log_additional_performance_metrics(
        self,
        pred: Any,
        data: dict[str, Tensor],
        stage: str,
    ) -> None:
        return

    def training_step(self, data, batch_idx):
        pred = self.forward(data)
        loss = self.nll(pred, data)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_additional_performance_metrics(pred, data, "train")
        return loss

    def validation_step(self, data, batch_idx):
        pred = self.forward(data)
        loss = self.nll(pred, data)
        self.log("val_loss", loss, prog_bar=True)
        self.log_additional_performance_metrics(pred, data, "val")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


##############################################################
######################## Subclasses ##########################
##############################################################


class FullTrainingSentenceDecoder(FullTraining):
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
    ) -> FloatTensor:
        z_true = data["z"]
        if self.hparams.discrete_z:
            z_logits = pred
            dist = torch.distributions.Categorical(logits=z_logits)
            logp = dist.log_prob(z_true)
        else:
            z_mu, z_logstd = pred
            logp = torch.distributions.Normal(z_mu, z_logstd.exp()).log_prob(z_true)
        logp = logp.mean()
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
            f"{stage}_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class FullTrainingHuggingFaceSentence(FullTraining):
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

        if learn_embeddings:
            model[0].auto_model = AutoModel.from_config(self.config)
        else:
            model[0].auto_model.embeddings.requires_grad_(False)

        super().__init__(*args, model=model, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        
        if self.hparams.reset_model_params:
            self.reset_model_params()

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

    def forward(
        self, data: dict[str, Tensor]
    ) -> FloatTensor | tuple[FloatTensor, FloatTensor]:
        if self.hparams.learn_embeddings:
            w: LongTensor = data["w_short"]
        else:
            w: LongTensor = data["w"]
        attention_mask = torch.ones_like(w)
        attention_mask[w == 1] = 0  # Assumes 1 is the padding token
        w = {"input_ids": w, "attention_mask": attention_mask}
        z_mu = self.model.forward(w)["sentence_embedding"]
        z_logstd = math.log(1.0) * torch.ones_like(z_mu)
        return z_mu, z_logstd

    def nll(
        self,
        pred: FloatTensor | tuple[FloatTensor, FloatTensor],
        data: dict[str, Tensor],
    ) -> FloatTensor:
        z_true = data["z"]
        z_mu, z_logstd = pred
        logp = torch.distributions.Normal(z_mu, z_logstd.exp()).log_prob(z_true)
        logp = logp.mean()
        return -logp
