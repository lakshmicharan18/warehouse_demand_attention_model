"""Simple no-training demand forecasting baselines."""

import numpy as np


def last_observation_baseline(X: np.ndarray) -> np.ndarray:
    """Forecast the next hour as the most recent observation in each window."""
    return np.asarray(X)[:, -1].copy()


def moving_average_baseline(X: np.ndarray) -> np.ndarray:
    """Forecast the next hour as the average of each 24-hour input window."""
    return np.asarray(X).mean(axis=1)
