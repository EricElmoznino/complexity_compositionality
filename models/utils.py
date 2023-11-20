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


class MLP(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_layers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers = []
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(num_inputs, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            num_inputs = hidden_dim
        layers.append(nn.Linear(num_inputs, num_outputs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.mlp(x)
