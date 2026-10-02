"""Frozen-model evaluation on the configured warehouse distribution shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from warehouse_demand_attention.baselines import last_observation_baseline, moving_average_baseline
from warehouse_demand_attention.data_generator import WarehouseDemandConfig, create_supervised_sequences, chronological_split, generate_warehouse_demand, shifted_distribution_config
from warehouse_demand_attention.training import DemandNormalizer, WarehouseTrainingResult, mae_rmse


@dataclass(frozen=True)
class DatasetStatistics:
    mean: float; std: float; minimum: float; maximum: float
    spike_count: int; drop_count: int


@dataclass(frozen=True)
class EvaluationMetrics:
    mae: float; rmse: float


@dataclass(frozen=True)
class GeneralizationResult:
    normal: dict[str, EvaluationMetrics]
    shifted: dict[str, EvaluationMetrics]
    normal_statistics: DatasetStatistics
    shifted_statistics: DatasetStatistics
    normal_config: WarehouseDemandConfig
    shifted_config: WarehouseDemandConfig
    shifted_targets: np.ndarray
    shifted_attention_predictions: np.ndarray
    shifted_last_predictions: np.ndarray
    shifted_attention_weights: np.ndarray


def dataset_statistics(config: WarehouseDemandConfig) -> DatasetStatistics:
    data = generate_warehouse_demand(config)
    return DatasetStatistics(float(data.demand.mean()), float(data.demand.std()), float(data.demand.min()), float(data.demand.max()), int(data.spike_events.sum()), int(data.drop_events.sum()))


def evaluate_frozen_model(normal_training: WarehouseTrainingResult) -> GeneralizationResult:
    """Evaluate trained normal-model weights on normal and shifted chronological tests."""
    normal_config = WarehouseDemandConfig()
    shifted_config = shifted_distribution_config(normal_config)
    normal_X = normal_training.data.normalizer.inverse_transform(normal_training.data.test_X)
    normal_y = normal_training.data.test_y
    normal_predictions = normal_training.test_predictions
    normal_metrics = _all_metrics(normal_X, normal_y, normal_predictions)

    shifted_data = generate_warehouse_demand(shifted_config)
    shifted_X, shifted_y = create_supervised_sequences(shifted_data.demand)
    shifted_split = chronological_split(shifted_X, shifted_y)
    raw_X, raw_y = shifted_split.X_test, shifted_split.y_test
    # Crucially, use the original normal-training statistics; do not fit on shifted data.
    normalized_X = normal_training.data.normalizer.transform(raw_X)
    normal_training.model.eval()
    with torch.no_grad():
        prediction, weights = normal_training.model(torch.tensor(normalized_X, dtype=torch.float32))
    shifted_predictions = normal_training.data.normalizer.inverse_transform(prediction.numpy())
    shifted_metrics = _all_metrics(raw_X, raw_y, shifted_predictions)
    return GeneralizationResult(
        normal_metrics, shifted_metrics, dataset_statistics(normal_config), dataset_statistics(shifted_config),
        normal_config, shifted_config, raw_y, shifted_predictions, last_observation_baseline(raw_X), weights[:, -1].numpy(),
    )


def _all_metrics(X: np.ndarray, y: np.ndarray, attention_prediction: np.ndarray) -> dict[str, EvaluationMetrics]:
    return {
        "Last observation": EvaluationMetrics(*mae_rmse(last_observation_baseline(X), y)),
        "24-hour moving average": EvaluationMetrics(*mae_rmse(moving_average_baseline(X), y)),
        "Attention model": EvaluationMetrics(*mae_rmse(attention_prediction, y)),
    }
