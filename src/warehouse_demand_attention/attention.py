"""First-principles single-head scaled dot-product self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class ScaledDotProductSelfAttention(nn.Module):
    """Learned single-head self-attention for batched sequence inputs.

    Inputs have shape ``[batch_size, sequence_length, input_dim]``. Queries (Q)
    describe what each position looks for, keys (K) describe what each position
    offers, and values (V) are the information to combine. ``Q @ K^T`` compares
    every query with every key. Dividing by ``sqrt(d_k)`` keeps score magnitudes
    controlled before softmax converts them into weights. The final weighted sum,
    ``attention_weights @ V``, produces one context-aware value per position.
    """

    def __init__(self, input_dim: int, d_k: int, d_v: int, scale_scores: bool = True) -> None:
        super().__init__()
        if input_dim <= 0 or d_k <= 0 or d_v <= 0:
            raise ValueError("input_dim, d_k, and d_v must be positive")

        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v
        self.scale_scores = scale_scores

        self.W_Q = nn.Parameter(torch.empty(input_dim, d_k))
        self.W_K = nn.Parameter(torch.empty(input_dim, d_k))
        self.W_V = nn.Parameter(torch.empty(input_dim, d_v))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Xavier initialization to keep initial projection scales reasonable."""
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(output, attention_weights)`` for ``[batch, sequence, features]`` X.

        The output shape is ``[batch_size, sequence_length, d_v]`` and attention
        weights have shape ``[batch_size, sequence_length, sequence_length]``.
        """
        output, attention_weights, _ = self.forward_with_details(X)
        return output, attention_weights

    def forward_with_details(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run attention and expose named intermediate tensors for inspection/tests."""
        if X.ndim != 3:
            raise ValueError("X must have shape [batch_size, sequence_length, input_dim]")
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {X.shape[-1]}")

        # Q, K, and V each retain batch and sequence dimensions after projection.
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Each query position is compared against every key position.
        scores = Q @ K.transpose(-2, -1)
        scaled_scores = scores / math.sqrt(self.d_k) if self.scale_scores else scores

        # Explicit stable softmax over key positions for every query row.
        stable_scores = scaled_scores - scaled_scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(stable_scores)
        attention_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

        # Each position receives the attention-weighted combination of all values.
        output = attention_weights @ V
        details = {
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "scaled_scores": scaled_scores,
        }
        return output, attention_weights, details
