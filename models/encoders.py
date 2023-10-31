from abc import ABC, abstractmethod
from typing import Literal
import math
import torch
from torch import nn, FloatTensor
from models.utils import learned_token_encodings, positional_token_encodings


class VqVaeEncoder(ABC, nn.Module):
    def __init__(self, emb_dim: int, num_words: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_words = num_words

    @abstractmethod
    def forward(self, z: FloatTensor) -> FloatTensor:
        """
        Args:
            z (FloatTensor): (bs, ...)

        Returns:
            FloatTensor: (bs, num_words, emb_dim)
        """
        pass


##############################################################
############# Subclasses for different encoders ##############
##############################################################


class TransformerVqVaeEncoder(VqVaeEncoder):
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
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        assert repr_dim >= emb_dim, "repr_dim must be >= emb_dim"
        self.repr_dim = repr_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim = emb_dim if mlp_hidden_dim is None else mlp_hidden_dim
        self.dropout = dropout
        self.token_encoding_type = token_encoding_type

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

    def forward(self, z: FloatTensor) -> FloatTensor:
        bs = z.shape[0]

        # Reshape z to (bs, num_repr_tokens, emb_dim)
        assert (
            z.ndim == 2 and z.shape[1] == self.repr_dim
        ), f"z must be (bs, {self.repr_dim}), got {z.shape}"
        if self.leftover_repr_dim > 0:
            z = torch.cat([z, z.new_zeros(bs, self.leftover_repr_dim)], dim=1)
        z = z.reshape(bs, self.num_repr_tokens, self.emb_dim)

        # Build transformer input
        input = z + self.repr_token_encoding
        input = torch.cat([self.token_encoding.repeat(bs, 1, 1), input], dim=1)

        # Get final token embeddings
        w_emb = self.transformer(input)[:, : self.num_words, :]

        return w_emb
