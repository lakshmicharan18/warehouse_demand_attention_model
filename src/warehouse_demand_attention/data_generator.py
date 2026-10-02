"""Synthetic hourly warehouse-demand data for forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WarehouseDemandConfig:
    """Parameters controlling the synthetic warehouse environment."""

    num_days: int = 180
    seed: int = 42
    base_demand: float = 125.0
    noise_std: float = 8.0
    spike_probability: float = 0.015
    drop_probability: float = 0.01
    spike_magnitude_range: Tuple[float, float] = (30.0, 65.0)
    drop_magnitude_range: Tuple[float, float] = (25.0, 55.0)
    start_datetime: datetime = datetime(2024, 1, 1)

    # These weights sum to less than one, keeping the autoregressive process stable.
    seasonal_weight: float = 0.55
    lag_1_weight: float = 0.20
    lag_2_weight: float = 0.10
    lag_24_weight: float = 0.15


@dataclass(frozen=True)
class GeneratedDemand:
    """Hourly demand values and the event indicators that produced them."""

    timestamps: np.ndarray
    demand: np.ndarray
    spike_events: np.ndarray
    drop_events: np.ndarray

    @property
    def hours(self) -> np.ndarray:
        return np.array([timestamp.hour for timestamp in self.timestamps], dtype=np.int64)

    @property
    def day_of_week(self) -> np.ndarray:
        return np.array([timestamp.weekday() for timestamp in self.timestamps], dtype=np.int64)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Non-overlapping supervised samples kept in chronological order."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


# Percent of base demand by hour: quiet overnight, rising morning, busy daytime,
# evening peak, then a late-night decline.
DAILY_PROFILE = np.array(
    [0.55, 0.50, 0.48, 0.47, 0.48, 0.53, 0.65, 0.80, 0.95, 1.05, 1.10, 1.12,
     1.10, 1.08, 1.10, 1.13, 1.20, 1.30, 1.35, 1.28, 1.10, 0.90, 0.72, 0.62],
    dtype=np.float64,
)
WEEKDAY_MULTIPLIERS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.78], dtype=np.float64)


def generate_warehouse_demand(config: WarehouseDemandConfig | None = None) -> GeneratedDemand:
    """Generate deterministic-for-seed, hourly warehouse demand observations."""
    config = config or WarehouseDemandConfig()
    if config.num_days <= 0:
        raise ValueError("num_days must be positive")
    if config.noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    _validate_probability(config.spike_probability, "spike_probability")
    _validate_probability(config.drop_probability, "drop_probability")
    _validate_range(config.spike_magnitude_range, "spike_magnitude_range")
    _validate_range(config.drop_magnitude_range, "drop_magnitude_range")

    total_hours = config.num_days * 24
    timestamps = np.array(
        [config.start_datetime + timedelta(hours=index) for index in range(total_hours)],
        dtype=object,
    )
    rng = np.random.default_rng(config.seed)
    demand = np.zeros(total_hours, dtype=np.float64)
    spike_events = rng.random(total_hours) < config.spike_probability
    drop_events = rng.random(total_hours) < config.drop_probability

    for index, timestamp in enumerate(timestamps):
        seasonal = (
            config.base_demand
            * DAILY_PROFILE[timestamp.hour]
            * WEEKDAY_MULTIPLIERS[timestamp.weekday()]
        )
        # Use the seasonal level for unavailable history at the beginning.
        lag_1 = demand[index - 1] if index >= 1 else seasonal
        lag_2 = demand[index - 2] if index >= 2 else seasonal
        lag_24 = demand[index - 24] if index >= 24 else seasonal

        event = 0.0
        if spike_events[index]:
            event += rng.uniform(*config.spike_magnitude_range)
        if drop_events[index]:
            event -= rng.uniform(*config.drop_magnitude_range)

        demand[index] = max(
            0.0,
            config.seasonal_weight * seasonal
            + config.lag_1_weight * lag_1
            + config.lag_2_weight * lag_2
            + config.lag_24_weight * lag_24
            + event
            + rng.normal(0.0, config.noise_std),
        )

    return GeneratedDemand(timestamps, demand, spike_events, drop_events)


def create_supervised_sequences(demand: np.ndarray, window_size: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Turn a one-dimensional demand series into prior-window/next-hour samples."""
    values = np.asarray(demand, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("demand must be one-dimensional")
    if window_size <= 0 or len(values) <= window_size:
        raise ValueError("demand must contain more values than window_size")

    X = np.stack([values[index:index + window_size] for index in range(len(values) - window_size)])
    y = values[window_size:].copy()
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split already ordered supervised samples without shuffling."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test set")

    train_end = int(len(X) * train_fraction)
    validation_end = train_end + int(len(X) * validation_fraction)
    return ChronologicalSplit(
        X_train=X[:train_end], y_train=y[:train_end],
        X_validation=X[train_end:validation_end], y_validation=y[train_end:validation_end],
        X_test=X[validation_end:], y_test=y[validation_end:],
    )


def shifted_distribution_config(config: WarehouseDemandConfig | None = None) -> WarehouseDemandConfig:
    """Return a configuration for a later robustness experiment, without running it."""
    base = config or WarehouseDemandConfig()
    return replace(
        base,
        noise_std=base.noise_std * 1.5,
        spike_probability=min(1.0, base.spike_probability * 2),
        spike_magnitude_range=(base.spike_magnitude_range[0] * 1.25, base.spike_magnitude_range[1] * 1.5),
    )


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _validate_range(value: Tuple[float, float], name: str) -> None:
    if value[0] < 0 or value[0] > value[1]:
        raise ValueError(f"{name} must be an ordered, non-negative range")
