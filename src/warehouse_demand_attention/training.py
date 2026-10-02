"""Chronological data preparation and CPU training for warehouse forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from warehouse_demand_attention.data_generator import create_supervised_sequences, chronological_split, generate_warehouse_demand
from warehouse_demand_attention.warehouse_model import WarehouseDemandAttentionModel


@dataclass(frozen=True)
class DemandNormalizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, train_X: np.ndarray) -> "DemandNormalizer":
        return cls(float(train_X.mean()), max(float(train_X.std()), 1e-8))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class WarehouseTrainingConfig:
    embedding_dim: int = 32
    d_k: int = 32
    d_v: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 80
    seed: int = 42


@dataclass(frozen=True)
class PreparedWarehouseData:
    train_X: np.ndarray; train_y: np.ndarray
    validation_X: np.ndarray; validation_y: np.ndarray
    test_X: np.ndarray; test_y: np.ndarray
    normalizer: DemandNormalizer


@dataclass(frozen=True)
class WarehouseTrainingResult:
    model: WarehouseDemandAttentionModel
    data: PreparedWarehouseData
    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    test_predictions: np.ndarray
    final_attention_weights: np.ndarray


def prepare_warehouse_data() -> PreparedWarehouseData:
    """Generate then chronologically split data before fitting train-only scaling."""
    generated = generate_warehouse_demand()
    X, y = create_supervised_sequences(generated.demand)
    split = chronological_split(X, y)
    normalizer = DemandNormalizer.fit(split.X_train)
    return PreparedWarehouseData(
        normalizer.transform(split.X_train), normalizer.transform(split.y_train),
        normalizer.transform(split.X_validation), normalizer.transform(split.y_validation),
        normalizer.transform(split.X_test), split.y_test.copy(), normalizer,
    )


def train_warehouse_model(config: WarehouseTrainingConfig | None = None) -> WarehouseTrainingResult:
    config = config or WarehouseTrainingConfig()
    data = prepare_warehouse_data()
    torch.manual_seed(config.seed)
    model = WarehouseDemandAttentionModel(24, config.embedding_dim, config.d_k, config.d_v)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(data.train_X, dtype=torch.float32), torch.tensor(data.train_y, dtype=torch.float32)), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed))
    train_losses, validation_losses = [], []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    validation_X = torch.tensor(data.validation_X, dtype=torch.float32)
    validation_y = torch.tensor(data.validation_y, dtype=torch.float32)
    for epoch in range(config.epochs):
        model.train(); total = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad(); prediction, _ = model(X_batch)
            loss = loss_fn(prediction, y_batch); loss.backward(); optimizer.step()
            total += loss.item() * len(X_batch)
        train_losses.append(total / len(data.train_y))
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(validation_X)[0], validation_y).item()
        validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch + 1, deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        normalized_prediction, weights = model(torch.tensor(data.test_X, dtype=torch.float32))
    return WarehouseTrainingResult(model, data, train_losses, validation_losses, best_epoch, data.normalizer.inverse_transform(normalized_prediction.numpy()), weights[:, -1].numpy())


def mae_rmse(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    errors = np.asarray(predictions) - np.asarray(targets)
    return float(np.abs(errors).mean()), float(np.sqrt(np.mean(errors ** 2)))
