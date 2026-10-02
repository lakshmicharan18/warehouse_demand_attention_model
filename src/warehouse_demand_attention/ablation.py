"""Matched scaled-versus-unscaled attention experiment on the toy task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from warehouse_demand_attention.toy_task import (
    ToyTaskConfig,
    ToyTrainingConfig,
    make_toy_datasets,
    train_toy_model,
)


@dataclass(frozen=True)
class AblationRunMetrics:
    seed: int
    initial_training_loss: float
    final_training_loss: float
    validation_loss: float
    test_mse: float
    test_mae: float
    mean_max_attention_weight: float
    attention_entropy: float
    mean_absolute_logit: float
    max_absolute_logit: float
    mean_gradient_norm: float
    gradients_finite: bool
    training_losses: tuple[float, ...]


@dataclass(frozen=True)
class AblationSummary:
    scaled_runs: tuple[AblationRunMetrics, ...]
    unscaled_runs: tuple[AblationRunMetrics, ...]


def run_attention_ablation(
    seeds: Iterable[int] = (42, 123, 2026), epochs: int = 50
) -> AblationSummary:
    """Run matched toy-task trials; scaling is the only variant-specific setting."""
    task_config = ToyTaskConfig()
    scaled_runs = []
    unscaled_runs = []
    for seed in seeds:
        shared = dict(
            embedding_dim=32,
            d_k=32,
            d_v=32,
            learning_rate=1e-3,
            batch_size=128,
            epochs=epochs,
            model_seed=seed,
        )
        scaled_runs.append(_run_variant(task_config, ToyTrainingConfig(**shared, scale_scores=True), seed))
        unscaled_runs.append(_run_variant(task_config, ToyTrainingConfig(**shared, scale_scores=False), seed))
    return AblationSummary(tuple(scaled_runs), tuple(unscaled_runs))


def metric_mean_and_std(runs: tuple[AblationRunMetrics, ...], metric: str) -> tuple[float, float]:
    values = np.array([getattr(run, metric) for run in runs], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _run_variant(
    task_config: ToyTaskConfig, training_config: ToyTrainingConfig, seed: int
) -> AblationRunMetrics:
    result = train_toy_model(task_config, training_config)
    datasets = make_toy_datasets(task_config)
    test_X = datasets.test.tensors[0]
    result.model.eval()
    with torch.no_grad():
        _, weights, details = result.model.forward_with_attention_details(test_X)
        probabilities = weights.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean().item()
        logits = details["scaled_scores"]

    return AblationRunMetrics(
        seed=seed,
        initial_training_loss=result.training_losses[0],
        final_training_loss=result.training_losses[-1],
        validation_loss=result.validation_losses[-1],
        test_mse=result.test_mse,
        test_mae=result.test_mae,
        mean_max_attention_weight=weights.max(dim=-1).values.mean().item(),
        attention_entropy=entropy,
        mean_absolute_logit=logits.abs().mean().item(),
        max_absolute_logit=logits.abs().max().item(),
        mean_gradient_norm=result.mean_gradient_norm,
        gradients_finite=result.gradients_finite,
        training_losses=tuple(result.training_losses),
    )
