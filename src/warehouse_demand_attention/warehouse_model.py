"""Small next-hour warehouse-demand forecaster using custom self-attention."""

import torch
from torch import nn

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention


class WarehouseDemandAttentionModel(nn.Module):
    """24 values -> projection + positions -> attention -> final context -> demand.

    The final input position is the most recent hour. Its attention output can
    combine all historical positions; no causal mask is needed because the target
    hour is outside the 24-hour input window.
    """

    def __init__(self, sequence_length: int = 24, embedding_dim: int = 32, d_k: int = 32, d_v: int = 32) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(1, embedding_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(sequence_length, embedding_dim))
        nn.init.normal_(self.position_embeddings, std=0.02)
        self.attention = ScaledDotProductSelfAttention(embedding_dim, d_k, d_v)
        self.prediction_head = nn.Linear(d_v, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if X.ndim != 2 or X.shape[1] != self.sequence_length:
            raise ValueError(f"X must have shape [batch_size, {self.sequence_length}]")
        encoded = self.input_projection(X.unsqueeze(-1)) + self.position_embeddings
        attended, weights = self.attention(encoded)
        return self.prediction_head(attended[:, -1]).squeeze(-1), weights
