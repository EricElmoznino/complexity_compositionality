from typing import Literal
import torch
from torch import nn, FloatTensor, LongTensor
from torch.nn import functional as F
from lightning import LightningModule
import wandb
import numpy as np
from sklearn.cluster import KMeans
from models.encoders import EmbeddingEncoder
from models.decoders import EmbeddingDecoder
from models.language_models import EmbeddingLM


class VqVae(LightningModule):
    DiscretizerType = Literal["straight_through", "gumbel_softmax"]

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        decoder: EmbeddingDecoder,
        lm: EmbeddingLM,
        vocab_size: int,
        beta_lm: float = 1.0,
        beta_codebook: float = 1.0,
        beta_commit: float = 0.25,
        lr: float = 1e-3,
        lr_lm: float | None = None,
        lr_discretizer: float | None = None,
        t_init: int = 0,
        t_reestimate: int = 10,
        p_reestimate: int = 2,
        t_lm: int = 20,
        encoder_norm: bool = True,
        discretizer: str = "straight_through",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder", "lm"])
        self.encoder = encoder
        self.decoder = decoder
        self.lm = lm
        if discretizer == "straight_through":
            self.discretizer = StraightThroughDiscretizer(
                self.hparams.vocab_size, self.encoder.emb_dim
            )
        else:
            self.discretizer = GumbelSoftmaxDiscretizer(
                self.hparams.vocab_size, self.encoder.emb_dim
            )
        if self.hparams.encoder_norm:
            self.encoder_norm = nn.BatchNorm1d(self.encoder.emb_dim)
        else:
            self.encoder_norm = nn.Identity()

        # At different stages of training, we want to enable/disable different parts
        self.use_discretizer = True
        self.use_lm = True
        self.fit_autoencoder = True

        # For word usage statistics
        self.train_word_usage = torch.zeros(
            self.encoder.num_words, vocab_size, dtype=torch.int64
        )
        self.val_word_usage = torch.zeros(
            self.encoder.num_words, vocab_size, dtype=torch.int64
        )

    def training_step(self, z: FloatTensor, batch_idx: int) -> FloatTensor:
        # Get predictions
        z_e = self.encoder(z)
        z_e = self.encoder_norm(z_e.transpose(1, 2)).transpose(1, 2)
        w, w_emb, loss_codebook, loss_commit = self.discretizer(z_e)
        if not self.use_discretizer:
            w_emb = z_e
        if not self.fit_autoencoder:
            w_emb = w_emb.detach()
        if self.use_lm:
            w_logits = self.lm(w_emb, self.discretizer.emb_table)
        if self.fit_autoencoder:
            zpred_mu, zpred_logstd = self.decoder(w_emb)

        # Compute losses
        losses = {
            "loss_recon": (
                self.k_z_given_w_and_decoder(z, zpred_mu, zpred_logstd, per_dim=True)
                if self.fit_autoencoder
                else 0.0
            ),
            "loss_lm": (
                self.k_w_given_language(w_logits, w, per_dim=True)
                if self.use_lm
                else 0.0
            ),
            "loss_codebook": (
                loss_codebook if self.use_discretizer and self.fit_autoencoder else 0.0
            ),
            "loss_commit": (
                loss_commit if self.use_discretizer and self.fit_autoencoder else 0.0
            ),
        }
        loss = (
            losses["loss_recon"]
            + self.hparams.beta_lm * losses["loss_lm"]
            + self.hparams.beta_codebook * losses["loss_codebook"]
            + self.hparams.beta_commit * losses["loss_commit"]
        )

        # Log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)
        if self.fit_autoencoder:
            self.log(
                "train/K(z|w,decoder)",
                self.k_z_given_w_and_decoder(z, zpred_mu, zpred_logstd),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train/MSE(z,z_mu)",
                F.mse_loss(z, zpred_mu),
                on_step=False,
                on_epoch=True,
            )
        if self.use_lm:
            self.log(
                "train/K(w|language)",
                self.k_w_given_language(w_logits, w),
                on_step=False,
                on_epoch=True,
            )
        self.train_word_usage += F.one_hot(w, self.hparams.vocab_size).sum(dim=0)

        return loss

    def validation_step(self, z: FloatTensor, batch_idx: int) -> None:
        # Get predictions
        z_e = self.encoder(z)
        z_e = self.encoder_norm(z_e.transpose(1, 2)).transpose(1, 2)
        w, w_emb, loss_codebook, loss_commit = self.discretizer(z_e)
        if not self.use_discretizer:
            w_emb = z_e
        if self.use_lm:
            w_logits = self.lm(w_emb, self.discretizer.emb_table)
        zpred_mu, zpred_logstd = self.decoder(w_emb)

        # Compute losses
        losses = {
            "loss_recon": self.k_z_given_w_and_decoder(
                z, zpred_mu, zpred_logstd, per_dim=True
            ),
            "loss_lm": (
                self.k_w_given_language(w_logits, w, per_dim=True)
                if self.use_lm
                else 0.0
            ),
            "loss_codebook": (
                loss_codebook if self.use_discretizer and self.fit_autoencoder else 0.0
            ),
            "loss_commit": (
                loss_commit if self.use_discretizer and self.fit_autoencoder else 0.0
            ),
        }
        loss = (
            losses["loss_recon"]
            + self.hparams.beta_lm * losses["loss_lm"]
            + self.hparams.beta_codebook * losses["loss_codebook"]
            + self.hparams.beta_commit * losses["loss_commit"]
        )

        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        for k, v in losses.items():
            self.log(f"val/{k}", v)
        if self.fit_autoencoder:
            self.log(
                "val/K(z|w,decoder)",
                self.k_z_given_w_and_decoder(z, zpred_mu, zpred_logstd),
            )
            self.log("val/MSE(z,z_mu)", F.mse_loss(z, zpred_mu))
        if self.use_lm:
            self.log("val/K(w|language)", self.k_w_given_language(w_logits, w))
        self.val_word_usage += F.one_hot(w, self.hparams.vocab_size).sum(dim=0)

        return loss

    def on_fit_start(self) -> None:
        self.train_word_usage = self.train_word_usage.to(self.device)
        self.val_word_usage = self.val_word_usage.to(self.device)

    def on_train_epoch_start(self) -> None:
        self.set_training_phase()
        if (
            self.hparams.t_init
            <= self.current_epoch
            < self.hparams.t_init + self.hparams.t_reestimate
        ):
            if (
                self.current_epoch - self.hparams.t_init
            ) % self.hparams.p_reestimate == 0 and self.current_epoch > 0:
                self.discretizer.emb_table.weight.data = (
                    self.reestimate_word_embeddings().clone()
                )

    def on_train_epoch_end(self) -> None:
        if self.logger is not None:
            self.log_word_usage("train")
        self.train_word_usage[:, :] = 0

    def on_validation_epoch_end(self) -> None:
        if self.logger is not None:
            self.log_word_usage("val")
        self.val_word_usage[:, :] = 0

    def k_z_given_w_and_decoder(
        self,
        z: FloatTensor,
        zpred_mu: FloatTensor,
        zpred_logstd: FloatTensor,
        per_dim: bool = False,
    ) -> torch.FloatTensor:
        dist = torch.distributions.Normal(zpred_mu, zpred_logstd.exp())
        k = -dist.log_prob(z)
        if per_dim:
            k = k.mean(dim=tuple(range(1, z.ndim)))
        else:
            k = k.sum(dim=tuple(range(1, z.ndim)))
        k = k.mean()
        return k

    def k_w_given_language(
        self,
        w_logits: FloatTensor,
        w: LongTensor,
        per_dim: bool = False,
    ) -> torch.FloatTensor:
        k = F.cross_entropy(w_logits.transpose(1, 2), w, reduction="none")
        if per_dim:
            k = k.mean(dim=-1)
        else:
            k = k.sum(dim=-1)
        k = k.mean()
        return k

    def set_training_phase(self) -> None:
        if self.current_epoch < self.hparams.t_init:
            self.use_discretizer = False
            self.use_lm = False
            self.fit_autoencoder = True
            self.log("misc/phase", 3.0, on_step=False, on_epoch=True)
        elif self.current_epoch < self.hparams.t_init + self.hparams.t_reestimate:
            self.use_discretizer = True
            self.use_lm = False
            self.fit_autoencoder = True
            self.log("misc/phase", 2.0, on_step=False, on_epoch=True)
        elif (
            self.current_epoch
            < self.hparams.t_init + self.hparams.t_reestimate + self.hparams.t_lm
        ):
            self.use_discretizer = True
            self.use_lm = True
            self.fit_autoencoder = False
            self.log("misc/phase", 1.0, on_step=False, on_epoch=True)
        else:
            self.use_discretizer = True
            self.use_lm = True
            self.fit_autoencoder = True
            self.log("misc/phase", 0.0, on_step=False, on_epoch=True)

    @torch.inference_mode()
    def reestimate_word_embeddings(self) -> torch.FloatTensor:
        z_es = []
        for z in self.trainer.datamodule.val_dataloader():
            z = z.to(self.device)
            z_e = self.encoder_norm(self.encoder(z).transpose(1, 2)).transpose(1, 2)
            z_es.append(z_e.cpu().numpy())
        z_es = np.concatenate(z_es, axis=0)
        z_es = z_es.reshape(-1, z_es.shape[-1])  # (bs * num_tokens, emb_dim)
        kmeans = KMeans(n_clusters=self.discretizer.vocab_size, n_init="auto")
        w_emb = kmeans.fit(z_es).cluster_centers_
        w_emb = torch.from_numpy(w_emb).float().to(self.device)
        return w_emb

    def log_word_usage(self, stage: str) -> None:
        data = self.train_word_usage if stage == "train" else self.val_word_usage
        data = data.float() / data.sum(dim=-1, keepdim=True)
        data = [
            [sent_pos, vocab_id, data[sent_pos, vocab_id].item()]
            for sent_pos in range(data.shape[0])
            for vocab_id in range(data.shape[1])
        ]
        table = wandb.Table(
            data=data, columns=["Sentence position", "Token ID", "Frequency"]
        )
        self.logger.experiment.log({"misc/word_frequencies": table})

    def configure_optimizers(self):
        lr_discretizer = (
            10 * self.hparams.lr
            if self.hparams.lr_discretizer is None
            else self.hparams.lr_discretizer
        )
        lr_lm = self.hparams.lr if self.hparams.lr_lm is None else self.hparams.lr_lm
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.encoder.parameters())
                    + list(self.decoder.parameters())
                    + list(self.encoder_norm.parameters()),
                },
                {"params": self.lm.parameters(), "lr": lr_lm},
                {"params": self.discretizer.parameters(), "lr": lr_discretizer},
            ],
            lr=self.hparams.lr,
        )
        return optimizer


class StraightThroughDiscretizer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_table = nn.Embedding(vocab_size, emb_dim)
        self.emb_table.weight.data.normal_()

    def forward(self, z_e: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        """
        Args:
            z_e (FloatTensor): (bs, num_tokens, emb_dim)

        Returns:
            tuple[LongTensor, FloatTensor, FloatTensor, FloatTensor]:
            w - (bs, num_tokens), w_emb - (bs, num_tokens, emb_dim),
            loss_codebook - (), loss_commit - ()
        """
        distances = torch.cdist(z_e, self.emb_table.weight)
        w = torch.argmin(distances, dim=-1)
        w_emb = self.emb_table(w)
        loss_codebook = F.mse_loss(w_emb, z_e.detach())
        loss_commit = F.mse_loss(z_e, w_emb.detach())
        w_emb = z_e + (w_emb - z_e).detach()  # For straight-through gradient estimator
        return w, w_emb, loss_codebook, loss_commit


class GumbelSoftmaxDiscretizer(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_table = nn.Embedding(vocab_size, emb_dim)
        self.emb_table.weight.data.normal_()

    def forward(self, z_e: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        """
        Args:
            z_e (FloatTensor): (bs, num_tokens, emb_dim)

        Returns:
            tuple[LongTensor, FloatTensor, FloatTensor, FloatTensor]:
            w - (bs, num_tokens), w_emb - (bs, num_tokens, emb_dim),
            loss_codebook - (), loss_commit - ()
        """
        distances = torch.cdist(z_e, self.emb_table.weight)
        if self.training:
            w_onehot = F.gumbel_softmax(-distances, tau=1.0, hard=False, dim=-1)
            w = torch.argmax(w_onehot, dim=-1)
        else:
            w = torch.argmin(distances, dim=-1)
            w_emb = self.emb_table(w)
        w_emb = self.emb_table(w)
        loss_codebook = F.mse_loss(w_emb, z_e.detach())
        loss_commit = F.mse_loss(z_e, w_emb.detach())
        if self.training:
            w_emb = w_onehot @ self.emb_table.weight  # For gumbel gradient estimator
        return w, w_emb, loss_codebook, loss_commit
