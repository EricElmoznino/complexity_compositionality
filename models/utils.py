import math
import torch
from torch import nn, FloatTensor


def learned_token_encodings(emb_dim: int, num_tokens: int) -> nn.Parameter:
    """Learned positional encodings for Transformer tokens.

    Args:
        emb_dim (int): Embedding dimension.
        num_tokens (int): Number of tokens.

    Returns:
        nn.Parameter: (1, num_tokens, emb_dim)
    """
    pe = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, num_tokens, emb_dim)))
    return pe


def positional_token_encodings(emb_dim: int, num_tokens: int) -> FloatTensor:
    """Relative positional encodings for Transformer tokens.

    Args:
        emb_dim (int): Embedding dimension.
        num_tokens (int): Number of tokens.

    Returns:
        FloatTensor: (1, num_tokens, emb_dim)
    """
    assert emb_dim % 2 == 0, "emb_dim must be even"
    pe = torch.zeros(num_tokens, emb_dim)
    position = torch.arange(0, num_tokens).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe
