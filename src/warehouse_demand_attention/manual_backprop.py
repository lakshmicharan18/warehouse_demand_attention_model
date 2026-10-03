"""Explicit matrix-calculus backward pass for scaled dot-product attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ManualBackpropResult:
    manual: dict[str, torch.Tensor]
    autograd: dict[str, torch.Tensor]
    max_absolute_differences: dict[str, float]
    max_relative_differences: dict[str, float]


def deterministic_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixed [1, 3, 2] float64 example used for verification."""
    torch.manual_seed(314)
    X = torch.tensor([[[0.25, -0.5], [1.0, 0.75], [-0.25, 0.5]]], dtype=torch.float64)
    W_Q = torch.randn(2, 2, dtype=torch.float64) * 0.2
    W_K = torch.randn(2, 2, dtype=torch.float64) * 0.2
    W_V = torch.randn(2, 2, dtype=torch.float64) * 0.2
    return X, W_Q, W_K, W_V


def stable_softmax(scores: torch.Tensor) -> torch.Tensor:
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(shifted)
    return exp_scores / exp_scores.sum(dim=-1, keepdim=True)


def softmax_backward(attention: torch.Tensor, dA: torch.Tensor) -> torch.Tensor:
    """Apply each row Jacobian diag(a)-aa^T without constructing it explicitly."""
    return attention * (dA - (dA * attention).sum(dim=-1, keepdim=True))


def manual_attention_gradients(X: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor) -> dict[str, torch.Tensor]:
    """Manually differentiate mean(Y²) through scaled dot-product attention."""
    d_k = W_Q.shape[1]
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    A = stable_softmax(scores)
    Y = A @ V
    dY = 2 * Y / Y.numel()  # L = mean(Y²)
    dA = dY @ V.transpose(-2, -1)
    dV = A.transpose(-2, -1) @ dY
    d_scores = softmax_backward(A, dA)
    scale = math.sqrt(d_k)
    dQ = (d_scores @ K) / scale
    dK = (d_scores.transpose(-2, -1) @ Q) / scale
    dW_Q = (X.transpose(-2, -1) @ dQ).sum(dim=0)
    dW_K = (X.transpose(-2, -1) @ dK).sum(dim=0)
    dW_V = (X.transpose(-2, -1) @ dV).sum(dim=0)
    dX = dQ @ W_Q.transpose(-2, -1) + dK @ W_K.transpose(-2, -1) + dV @ W_V.transpose(-2, -1)
    return {"dW_Q": dW_Q, "dW_K": dW_K, "dW_V": dW_V, "dX": dX}


def compare_manual_to_autograd() -> ManualBackpropResult:
    """Compare manual matrix calculus with separate PyTorch autograd reference."""
    X, W_Q, W_K, W_V = deterministic_example()
    manual = manual_attention_gradients(X, W_Q, W_K, W_V)
    X_ref = X.detach().clone().requires_grad_(True)
    q_ref, k_ref, v_ref = [weight.detach().clone().requires_grad_(True) for weight in (W_Q, W_K, W_V)]
    Q, K, V = X_ref @ q_ref, X_ref @ k_ref, X_ref @ v_ref
    A = stable_softmax((Q @ K.transpose(-2, -1)) / math.sqrt(2))
    loss = (A @ V).square().mean()
    loss.backward()
    autograd = {"dW_Q": q_ref.grad, "dW_K": k_ref.grad, "dW_V": v_ref.grad, "dX": X_ref.grad}
    absolute, relative = {}, {}
    for name in manual:
        difference = (manual[name] - autograd[name]).abs()
        absolute[name] = difference.max().item()
        relative[name] = (difference / torch.maximum(manual[name].abs(), autograd[name].abs()).clamp_min(1e-12)).max().item()
    return ManualBackpropResult(manual, autograd, absolute, relative)
