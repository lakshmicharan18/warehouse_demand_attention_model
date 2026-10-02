"""Reproducible analysis of the maximum-error shifted warehouse forecast."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from warehouse_demand_attention.baselines import last_observation_baseline, moving_average_baseline
from warehouse_demand_attention.data_generator import create_supervised_sequences, chronological_split, generate_warehouse_demand, shifted_distribution_config
from warehouse_demand_attention.training import WarehouseTrainingResult


@dataclass(frozen=True)
class FailureCase:
    test_index: int; history: np.ndarray; actual: float; prediction: float; absolute_error: float
    last_prediction: float; moving_average_prediction: float; target_event: str
    history_spike_count: int; history_drop_count: int; attention_weights: np.ndarray


@dataclass(frozen=True)
class FailureAnalysis:
    case: FailureCase; absolute_errors: np.ndarray; spike_target_errors: np.ndarray; non_spike_target_errors: np.ndarray


def absolute_errors(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(actual) - np.asarray(prediction))


def largest_error_index(errors: np.ndarray) -> int:
    return int(np.argmax(errors))


def error_percentiles(errors: np.ndarray) -> dict[str, float]:
    return {"median": float(np.percentile(errors, 50)), "p90": float(np.percentile(errors, 90)), "p95": float(np.percentile(errors, 95)), "maximum": float(np.max(errors))}


def analyze_shifted_failure(trained: WarehouseTrainingResult) -> FailureAnalysis:
    """Evaluate frozen normal-trained weights and select the shifted maximum error."""
    data = generate_warehouse_demand(shifted_distribution_config())
    X, y = create_supervised_sequences(data.demand)
    split = chronological_split(X, y)
    test_X, test_y = split.X_test, split.y_test
    normalized_X = trained.data.normalizer.transform(test_X)
    trained.model.eval()
    with torch.no_grad():
        normalized_prediction, weights = trained.model(torch.tensor(normalized_X, dtype=torch.float32))
    prediction = trained.data.normalizer.inverse_transform(normalized_prediction.numpy())
    errors = absolute_errors(test_y, prediction)
    index = largest_error_index(errors)
    # Supervised target i corresponds to demand/event index i + 24.
    first_test_sample = len(split.X_train) + len(split.X_validation)
    target_indices = np.arange(first_test_sample, first_test_sample + len(test_y)) + 24
    target_index = int(target_indices[index])
    target_event = "spike" if data.spike_events[target_index] else "drop" if data.drop_events[target_index] else "normal"
    history_events = slice(target_index - 24, target_index)
    spike_targets = data.spike_events[target_indices]
    case = FailureCase(index, test_X[index], float(test_y[index]), float(prediction[index]), float(errors[index]), float(last_observation_baseline(test_X[index:index+1])[0]), float(moving_average_baseline(test_X[index:index+1])[0]), target_event, int(data.spike_events[history_events].sum()), int(data.drop_events[history_events].sum()), weights[index, -1].numpy())
    return FailureAnalysis(case, errors, errors[spike_targets], errors[~spike_targets])
