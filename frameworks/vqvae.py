import torch
from torch import nn, FloatTensor, LongTensor
from torch.nn import functional as F
from lightning import LightningModule
import numpy as np
from sklearn.cluster import KMeans
from models.encoders import VqVaeEncoder
from models.decoders import VqVaeDecoder
from models.language_models import VqVaeLM


class VqVae(LightningModule):
    def __init__(
        self,
        encoder: VqVaeEncoder,
        decoder: VqVaeDecoder,
        lm: VqVaeLM,
        vocab_size: int,
        beta_lm: float = 1.0,
        beta_codebook: float = 1.0,
        beta_commit: float = 1.0,
        lr: float = 1e-3,
        lr_lm: float | None = None,
        lr_discretizer: float | None = None,
        t_init: int = 0,
        t_reestimate: int = 10,
        p_reestimate: int = 2,
        t_lm: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder", "lm"])
        self.encoder = encoder
        self.decoder = decoder
        self.lm = lm
        self.discretizer = Discretizer(self.hparams.vocab_size, self.encoder.emb_dim)
        self.encoder_norm = nn.BatchNorm1d(self.encoder.emb_dim)

        # At different stages of training, we want to enable/disable different parts
        self.use_discretizer = True
        self.use_lm = True
        self.fit_autoencoder = True

    def training_step(self, z: FloatTensor, batch_idx: int) -> FloatTensor:
        # Get predictions
        z_e = self.encoder(z)
        z_e = self.encoder_norm(z_e.transpose(1, 2)).transpose(1, 2)
        if self.use_discretizer:
            w, w_emb = self.discretizer(z_e)
        else:
            w, w_emb = None, z_e
        if not self.fit_autoencoder:
            w_emb = w_emb.detach()
        if self.use_lm:
            w_logits = self.lm(w_emb, self.discretizer.emb_table)
        if self.fit_autoencoder:
            zpred_mu, zpred_logstd = self.decoder(w_emb)

        # Compute losses
        losses = {
            "loss_recon": self.k_z_given_w_and_decoder(
                z, zpred_mu, zpred_logstd, per_dim=True
            )
            if self.fit_autoencoder
            else 0.0,
            "loss_lm": self.k_w_given_language(w_logits, w, per_dim=True)
            if self.use_lm
            else 0.0,
            "loss_codebook": F.mse_loss(z_e, w_emb.detach())
            if self.use_discretizer and self.fit_autoencoder
            else 0.0,
            "loss_commit": F.mse_loss(w_emb, z_e.detach())
            if self.use_discretizer and self.fit_autoencoder
            else 0.0,
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
        if self.use_lm:
            self.log(
                "train/K(w|language)",
                self.k_w_given_language(w_logits, w),
                on_step=False,
                on_epoch=True,
            )

        return loss

    def validation_step(self, z: FloatTensor, batch_idx: int) -> None:
        # Get predictions
        z_e = self.encoder(z)
        z_e = self.encoder_norm(z_e.transpose(1, 2)).transpose(1, 2)
        if self.use_discretizer:
            w, w_emb = self.discretizer(z_e)
        else:
            w, w_emb = None, z_e
        if self.use_lm:
            w_logits = self.lm(w_emb, self.discretizer.emb_table)
        zpred_mu, zpred_logstd = self.decoder(w_emb)

        # Compute losses
        losses = {
            "loss_recon": self.k_z_given_w_and_decoder(
                z, zpred_mu, zpred_logstd, per_dim=True
            ),
            "loss_lm": self.k_w_given_language(w_logits, w, per_dim=True)
            if self.use_lm
            else 0.0,
            "loss_codebook": F.mse_loss(z_e, w_emb.detach())
            if self.use_discretizer
            else 0.0,
            "loss_commit": F.mse_loss(w_emb, z_e.detach())
            if self.use_discretizer
            else 0.0,
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
        if self.use_lm:
            self.log("val/K(w|language)", self.k_w_given_language(w_logits, w))

        return loss

    def on_train_epoch_start(self) -> None:
        self.set_training_phase()
        if (
            self.hparams.t_init
            <= self.current_epoch
            < self.hparams.t_init + self.hparams.t_reestimate
        ):
            if (
                self.current_epoch % self.hparams.p_reestimate == 0
                and self.current_epoch > 0
            ):
                self.reestimate_word_embeddings()

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
    def reestimate_word_embeddings(self) -> None:
        z_es = []
        for z in self.trainer.datamodule.val_dataloader():
            z = z.to(self.device)
            z_e = self.encoder_norm(self.encoder(z).transpose(1, 2)).transpose(1, 2)
            z_es.append(z_e.cpu().numpy())
        z_es = np.concatenate(z_es, axis=0)
        z_es = z_es.reshape(-1, z_es.shape[-1])  # (bs * num_tokens, emb_dim)
        kmeans = KMeans(n_clusters=self.discretizer.vocab_size, n_init="auto")
        w_emb = kmeans.fit(z_es).cluster_centers_
        w_emb = torch.from_numpy(w_emb).to(self.device)
        self.discretizer.emb_table.weight.data = w_emb

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


class Discretizer(nn.Module):
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
            tuple[LongTensor, FloatTensor]: w - (bs, num_tokens), w_emb - (bs, num_tokens, emb_dim)
        """
        distances = torch.cdist(z_e, self.emb_table.weight)
        w = torch.argmin(distances, dim=-1)
        w_emb = self.emb_table(w)
        w_emb = z_e + (w_emb - z_e).detach()  # For straight-through gradient estimator
        return w, w_emb
