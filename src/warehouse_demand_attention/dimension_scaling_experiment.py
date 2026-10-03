"""Controlled study of score scaling across query/key dimensions."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch

DIMENSIONS = (4, 8, 16, 32, 64, 128)
SEEDS = (11, 22, 33, 44, 55)

@dataclass(frozen=True)
class DimensionMetrics:
    d_k: int; raw_mean: float; raw_std: float; raw_mean_abs: float; raw_max_abs: float
    scaled_mean: float; scaled_std: float; scaled_mean_abs: float; scaled_max_abs: float
    unscaled_max_weight: float; scaled_max_weight: float; unscaled_entropy: float; scaled_entropy: float
    unscaled_gradient_norm: float; scaled_gradient_norm: float

def logits(Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = Q @ K.transpose(-2, -1)
    return raw, raw / math.sqrt(Q.shape[-1])

def attention_statistics(scores: torch.Tensor) -> tuple[float, float]:
    weights = torch.softmax(scores, dim=-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return weights.max(dim=-1).values.mean().item(), entropy.item()

def gradient_norm(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scaled: bool) -> float:
    query = Q.detach().clone().requires_grad_(True)
    scores = query @ K.transpose(-2, -1)
    if scaled: scores = scores / math.sqrt(query.shape[-1])
    output = torch.softmax(scores, dim=-1) @ V
    output.square().mean().backward()
    return query.grad.norm().item()

def run_trial(d_k: int, seed: int, batch_size: int = 128, sequence_length: int = 24) -> DimensionMetrics:
    torch.manual_seed(seed)
    Q = torch.randn(batch_size, sequence_length, d_k, dtype=torch.float64)
    K = torch.randn(batch_size, sequence_length, d_k, dtype=torch.float64)
    V = torch.randn(batch_size, sequence_length, d_k, dtype=torch.float64)
    raw, scaled = logits(Q, K)
    unscaled_weight, unscaled_entropy = attention_statistics(raw)
    scaled_weight, scaled_entropy = attention_statistics(scaled)
    return DimensionMetrics(d_k, raw.mean().item(), raw.std().item(), raw.abs().mean().item(), raw.abs().max().item(), scaled.mean().item(), scaled.std().item(), scaled.abs().mean().item(), scaled.abs().max().item(), unscaled_weight, scaled_weight, unscaled_entropy, scaled_entropy, gradient_norm(Q,K,V,False), gradient_norm(Q,K,V,True))

def run_experiment() -> list[DimensionMetrics]:
    results=[]
    for d_k in DIMENSIONS:
        trials=[run_trial(d_k, seed) for seed in SEEDS]
        values={field: float(np.mean([getattr(t, field) for t in trials])) for field in DimensionMetrics.__dataclass_fields__ if field != 'd_k'}
        results.append(DimensionMetrics(d_k=d_k, **values))
    return results
