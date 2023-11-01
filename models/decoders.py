from abc import ABC, abstractmethod
from typing import Literal
import math
import torch
from torch import nn, FloatTensor
from models.utils import learned_token_encodings, positional_token_encodings


class VqVaeDecoder(ABC, nn.Module):
    def __init__(self, emb_dim: int, num_words: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_words = num_words

    @abstractmethod
    def forward(self, w_emb: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        """
        Args:
            w_emb (FloatTensor): (bs, num_words, emb_dim)

        Returns:
            tuple[FloatTensor, FloatTensor]: mean - (bs, ...), logstd - (bs, ...)
        """
        pass


##############################################################
############# Subclasses for different decoders ##############
##############################################################


class TransformerVqVaeDecoder(VqVaeDecoder):
    TokenEncodingType = Literal["learned", "positional"]

    def __init__(
        self,
        emb_dim: int,
        num_words: int,
        repr_dim: int,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        token_encoding_type: TokenEncodingType = "learned",
        fixed_repr_std: float | None = 1.0,
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        assert repr_dim >= emb_dim, "repr_dim must be >= emb_dim"
        self.repr_dim = repr_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim = emb_dim if mlp_hidden_dim is None else mlp_hidden_dim
        self.dropout = dropout
        self.token_encoding_type = token_encoding_type
        self.fixed_repr_std = fixed_repr_std

        # Positional encodings
        if self.token_encoding_type == "learned":
            self.token_encoding = learned_token_encodings(emb_dim, num_words)
        elif self.token_encoding_type == "positional":
            self.token_encoding = positional_token_encodings(emb_dim, num_words)
        else:
            raise ValueError(
                f"token_encoding_type must be 'learned' or 'positional', got {token_encoding_type}"
            )
        self.num_repr_tokens = math.ceil(repr_dim / emb_dim)
        if fixed_repr_std is None:
            self.num_repr_tokens *= 2  # mean and std for each representation token
        self.leftover_repr_dim = emb_dim - repr_dim % emb_dim
        self.repr_token_encoding = learned_token_encodings(
            emb_dim, self.num_repr_tokens
        )

        # Transformer
        layer_spec = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=self.mlp_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer_spec, num_layers=num_layers
        )

        self.init_weights()

    def init_weights(self):
        # Xavier uniform init for the transformer
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, w_emb: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        bs = w_emb.shape[0]

        # Build transformer input
        input = w_emb + self.token_encoding
        input = torch.cat([self.repr_token_encoding.repeat(bs, 1, 1), input], dim=1)

        # Get final token embeddings
        z_mu = self.transformer(input)[:, : self.num_repr_tokens, :]
        if self.fixed_repr_std is None:
            z_mu, z_logstd = z_mu.chunk(2, dim=1)
        else:
            z_logstd = math.log(self.fixed_repr_std) * torch.ones_like(z_mu)

        # Reshape z to (bs, repr_dim)
        z_mu = z_mu.reshape(bs, self.repr_dim + self.leftover_repr_dim)
        z_logstd = z_logstd.reshape(bs, self.repr_dim + self.leftover_repr_dim)
        if self.leftover_repr_dim > 0:
            z_mu = z_mu[:, : -self.leftover_repr_dim]
            z_logstd = z_logstd[:, : -self.leftover_repr_dim]

        return z_mu, z_logstd
