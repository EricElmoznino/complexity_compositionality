from abc import ABC, abstractmethod
from typing import Literal
import math
import torch
from torch import nn, FloatTensor, LongTensor
from models.utils import learned_token_encodings, positional_token_encodings, MLP
from sentence_transformers import SentenceTransformer

##############################################################
#################### Embedding decoders ######################
##############################################################


class EmbeddingDecoder(ABC, nn.Module):
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


class TransformerEmbeddingDecoder(EmbeddingDecoder):
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
        min_std: float | None = 1e-2,
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        self.repr_dim = repr_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_hidden_dim = emb_dim if mlp_hidden_dim is None else mlp_hidden_dim
        self.dropout = dropout
        self.token_encoding_type = token_encoding_type
        self.fixed_repr_std = fixed_repr_std
        self.min_std = min_std

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
        self.leftover_repr_dim = (emb_dim - repr_dim % emb_dim) % emb_dim
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
            if self.min_std is not None:
                z_logstd = z_logstd.clamp(min=math.log(self.min_std))
        else:
            z_logstd = math.log(self.fixed_repr_std) * torch.ones_like(z_mu)

        # Reshape z to (bs, repr_dim)
        z_mu = z_mu.reshape(bs, self.repr_dim + self.leftover_repr_dim)
        z_logstd = z_logstd.reshape(bs, self.repr_dim + self.leftover_repr_dim)
        if self.leftover_repr_dim > 0:
            z_mu = z_mu[:, : -self.leftover_repr_dim]
            z_logstd = z_logstd[:, : -self.leftover_repr_dim]

        return z_mu, z_logstd


class MLPEmbeddingDecoder(EmbeddingDecoder):
    def __init__(
        self,
        emb_dim: int,
        num_words: int,
        repr_dim: int,
        num_layers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        fixed_repr_std: float | None = 1.0,
        min_std: float | None = 1e-2,
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        self.fixed_repr_std = fixed_repr_std
        self.min_std = min_std
        num_outputs = repr_dim * 2 if fixed_repr_std is None else repr_dim
        self.mlp = MLP(
            num_inputs=num_words * emb_dim,
            num_outputs=num_outputs,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, w_emb: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        w_emb = w_emb.reshape(w_emb.shape[0], -1)
        z_mu = self.mlp(w_emb)
        if self.fixed_repr_std is None:
            z_mu, z_logstd = z_mu.chunk(2, dim=1)
            if self.min_std is not None:
                z_logstd = z_logstd.clamp(min=math.log(self.min_std))
        else:
            z_logstd = math.log(self.fixed_repr_std) * torch.ones_like(z_mu)
        return z_mu, z_logstd


class CNNEmbeddingDecoder(EmbeddingDecoder):
    def __init__(
        self,
        emb_dim: int,
        num_words: int,
        cnn: nn.Module,
        fixed_repr_std: float | None = 1.0,
        w_spatial_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)
        self.cnn = cnn
        self.fixed_repr_std = fixed_repr_std
        self.w_spatial_shape = w_spatial_shape

    def forward(self, w_emb: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        if self.w_spatial_shape is None:
            h, w = int(self.num_words**0.5), int(self.num_words**0.5)
        else:
            h, w = self.w_spatial_shape
        w_emb = w_emb.transpose(1, 2)
        w_emb = w_emb.reshape(w_emb.shape[0], self.emb_dim, h, w)
        z_mu = self.cnn(w_emb)
        if self.fixed_repr_std is None:
            z_mu, z_logstd = z_mu.chunk(2, dim=1)
        else:
            z_logstd = math.log(self.fixed_repr_std) * torch.ones_like(z_mu)
        return z_mu, z_logstd


class IdentityEmbeddingDecoder(EmbeddingDecoder):
    def __init__(self, emb_dim: int, num_words: int) -> None:
        super().__init__(emb_dim=emb_dim, num_words=num_words)

    def forward(self, w_emb: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        w_emb = w_emb.reshape(w_emb.shape[0], -1)
        return w_emb, torch.zeros_like(w_emb)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class HuggingFaceEmbeddingDecoder():
    def __init__(self, emb_dim: int, fixed_repr_std: float):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device="cuda")
        self.model.apply(init_weights)
        self.emb_dim = emb_dim
        self.fixed_repr_std = fixed_repr_std

    def __call__(self, x):
        # x = self.model.tokenize(x)

        # build structure with input_ids and attention_mask
        # input_ids is just w 
        # attention_mask -> check documentation for attention_mask for SentenceTransformer -- maybe just set to 0 whenever padding token is 1
        # ask ChatGPT !
        attention_mask = torch.ones_like(x)
        attention_mask[x == 1] = 0
        x = {'input_ids': x, 'attention_mask': attention_mask}

        # x = {'input_ids': x['input_ids'].to('cuda'), 'attention_mask': x['attention_mask'].to('cuda')}
        out = self.model.forward(x)
        z_mu = out['sentence_embedding']
        z_logstd = math.log(self.fixed_repr_std) * torch.ones_like(z_mu)
        return z_mu, z_logstd

##############################################################
##################### Sentence decoders ######################
##############################################################


class SentenceDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_decoder: EmbeddingDecoder) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_decoder = embedding_decoder
        self.embedding = nn.Embedding(vocab_size, embedding_decoder.emb_dim)

    @abstractmethod
    def forward(self, w: LongTensor) -> tuple[FloatTensor, FloatTensor]:
        """
        Args:
            w (FloatTensor): (bs, num_words)

        Returns:
            tuple[FloatTensor, FloatTensor]: mean - (bs, ...), logstd - (bs, ...)
        """
        w_emb = self.embedding(w)
        return self.embedding_decoder(w_emb)

class HuggingFaceSentenceDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_decoder: HuggingFaceEmbeddingDecoder) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_decoder = embedding_decoder
        self.embedding = nn.Embedding(vocab_size, embedding_decoder.emb_dim)

    def forward(self, w: LongTensor) -> tuple[FloatTensor, FloatTensor]:
        return self.embedding_decoder(w)