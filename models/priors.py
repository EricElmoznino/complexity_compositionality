from abc import ABC, abstractmethod
from typing import Literal
import torch
from torch import nn, FloatTensor, LongTensor
from torch.nn import functional as F
from models.utils import learned_token_encodings, positional_token_encodings


class VQVAEPrior(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        w: LongTensor,
        w_emb: FloatTensor,
        w_emb_table: nn.Embedding,
        return_logits: bool = False,
    ) -> FloatTensor | tuple[FloatTensor, FloatTensor]:
        """
        Args:
            w (LongTensor): (bs, num_tokens)
            w_emb (FloatTensor): (bs, num_tokens, emb_dim)
            w_emb_table (nn.Embedding): Token embedding table
            return_logits (bool): Whether to return logits in addition to -logp(w)

        Returns:
            FloatTensor | tuple[FloatTensor, FloatTensor]: -logp(w) (i.e., cross-entropy loss), as well as w logits if return_logits is True
        """
        pass


##############################################################
############## Subclasses for different priors ###############
##############################################################


class TransformerVQVAEPrior(VQVAEPrior):
    TokenEncodingType = Literal["learned", "positional"]
    EmbeddingLookupType = Literal["linear", "distance"]

    def __init__(
        self,
        emb_dim: int,
        num_tokens: int,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        token_encoding_type: TokenEncodingType = "learned",
        embedding_lookup_type: EmbeddingLookupType = "linear",
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim = emb_dim if mlp_hidden_dim is None else mlp_hidden_dim
        self.dropout = dropout
        self.token_encoding_type = token_encoding_type
        self.embedding_lookup_type = embedding_lookup_type

        # Positional encodings (+1 for start token)
        if self.token_encoding_type == "learned":
            self.token_encoding = learned_token_encodings(emb_dim, num_tokens + 1)
        elif self.token_encoding_type == "positional":
            self.token_encoding = positional_token_encodings(emb_dim, num_tokens + 1)
        else:
            raise ValueError(
                f"token_encoding_type must be 'learned' or 'positional', got {token_encoding_type}"
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

    def forward(
        self,
        w: LongTensor,
        w_emb: FloatTensor,
        w_emb_table: nn.Embedding,
        return_logits: bool = False,
    ) -> FloatTensor:
        w_emb = torch.cat(
            [w_emb.new_zeros(w_emb.shape[0], 1, w_emb.shape[2]), w_emb], dim=1
        )
        w_emb += self.token_encoding
        out = self.transformer(w_emb)[:, :-1]
        out = out.unsqueeze(2)
        w_emb_table = w_emb_table.weight.unsqueeze(0).unsqueeze(0)
        if self.embedding_lookup_type == "linear":
            logits = (out * w_emb_table).sum(dim=-1)
        elif self.embedding_lookup_type == "distance":
            logits = -((out - w_emb_table) ** 2).sum(dim=-1).sqrt()
        else:
            raise ValueError(
                f"embedding_lookup_type must be 'linear' or 'distance', got {self.embedding_lookup_type}"
            )
        neg_logp = F.cross_entropy(logits.permute(0, 2, 1), w, reduction="none").sum(-1)
        if return_logits:
            return neg_logp, logits
        else:
            return neg_logp
