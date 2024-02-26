from abc import ABC, abstractmethod
from typing import Literal
import torch
from torch import nn, FloatTensor
from models.utils import learned_token_encodings, positional_token_encodings


##############################################################
####################### Embedding LMs ########################
##############################################################


class EmbeddingLM(ABC, nn.Module):
    def __init__(self, emb_dim: int, num_words: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_words = num_words

    @abstractmethod
    def forward(
        self,
        w_emb: FloatTensor,
        w_emb_table: nn.Embedding,
    ) -> FloatTensor:
        """
        Args:
            w_emb (FloatTensor): (bs, num_words, emb_dim)
            w_emb_table (nn.Embedding): Token embedding table
            return_logits (bool): Whether to return logits in addition to -logp(w)

        Returns:
            FloatTensor: (bs, num_words, vocab_size) logits for all words
        """
        pass


class TransformerEmbeddingLM(EmbeddingLM):
    TokenEncodingType = Literal["learned", "positional"]
    EmbeddingLookupType = Literal["linear", "distance"]

    def __init__(
        self,
        emb_dim: int,
        num_words: int,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        word_encoding_type: TokenEncodingType = "learned",
        embedding_lookup_type: EmbeddingLookupType = "linear",
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim = emb_dim if mlp_hidden_dim is None else mlp_hidden_dim
        self.dropout = dropout
        self.word_encoding_type = word_encoding_type
        self.embedding_lookup_type = embedding_lookup_type

        # Positional encodings (+1 for start word)
        if self.word_encoding_type == "learned":
            self.word_encoding = learned_token_encodings(emb_dim, num_words + 1)
        elif self.word_encoding_type == "positional":
            self.word_encoding = positional_token_encodings(emb_dim, num_words + 1)
        else:
            raise ValueError(
                f"word_encoding_type must be 'learned' or 'positional', got {word_encoding_type}"
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
        w_emb: FloatTensor,
        w_emb_table: nn.Embedding,
    ) -> FloatTensor:
        w_emb = torch.cat(
            [w_emb.new_zeros(w_emb.shape[0], 1, w_emb.shape[2]), w_emb], dim=1
        )
        w_emb += self.word_encoding
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
        return logits
