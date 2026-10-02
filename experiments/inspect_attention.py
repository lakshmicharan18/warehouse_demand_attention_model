"""Print a deterministic, inspectable first-principles attention forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    attention = ScaledDotProductSelfAttention(input_dim=2, d_k=2, d_v=2)
    output, weights, details = attention.forward_with_details(X)

    print(f"X shape: {tuple(X.shape)}")
    print(f"Q shape: {tuple(details['Q'].shape)}")
    print(f"K shape: {tuple(details['K'].shape)}")
    print(f"V shape: {tuple(details['V'].shape)}")
    print(f"Score shape: {tuple(details['scores'].shape)}")
    print(f"Attention-weight shape: {tuple(weights.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Attention weights:")
    print(weights[0])
    print("Row sums:")
    print(weights[0].sum(dim=-1))


if __name__ == "__main__":
    main()
