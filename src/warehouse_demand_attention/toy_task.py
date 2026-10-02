"""A small sequence-regression task for demonstrating custom attention learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention


@dataclass(frozen=True)
class ToyTaskConfig:
    sequence_length: int = 6
    train_samples: int = 2_000
    validation_samples: int = 400
    test_samples: int = 400
    seed: int = 42
    target_noise_std: float = 0.01


@dataclass(frozen=True)
class ToyTrainingConfig:
    embedding_dim: int = 8
    d_k: int = 8
    d_v: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    model_seed: int = 123
    scale_scores: bool = True


@dataclass(frozen=True)
class ToyDatasets:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset


@dataclass(frozen=True)
class ToyTrainingResult:
    model: "ToyAttentionRegressor"
    training_losses: list[float]
    validation_losses: list[float]
    test_mse: float
    test_mae: float
    gradients_finite: bool
    mean_gradient_norm: float


def generate_toy_data(
    num_samples: int, seed: int, sequence_length: int = 6, noise_std: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate scalar sequences and targets dominated by positions 1 and 4.

    Each input has shape ``[num_samples, 6]`` and values sampled uniformly from
    ``[-1, 1]``. The target is ``0.7 * x[1] + 0.3 * x[4] + noise``.
    """
    if sequence_length != 6:
        raise ValueError("the toy target is defined for sequences of length 6")
    if num_samples <= 0 or noise_std < 0:
        raise ValueError("num_samples must be positive and noise_std non-negative")

    generator = torch.Generator().manual_seed(seed)
    X = torch.rand(num_samples, sequence_length, generator=generator) * 2 - 1
    noise = torch.randn(num_samples, generator=generator) * noise_std
    y = 0.7 * X[:, 1] + 0.3 * X[:, 4] + noise
    return X, y


def make_toy_datasets(config: ToyTaskConfig | None = None) -> ToyDatasets:
    """Create deterministic, non-overlapping train/validation/test toy datasets."""
    config = config or ToyTaskConfig()
    train = TensorDataset(*generate_toy_data(
        config.train_samples, config.seed, config.sequence_length, config.target_noise_std
    ))
    validation = TensorDataset(*generate_toy_data(
        config.validation_samples, config.seed + 1, config.sequence_length, config.target_noise_std
    ))
    test = TensorDataset(*generate_toy_data(
        config.test_samples, config.seed + 2, config.sequence_length, config.target_noise_std
    ))
    return ToyDatasets(train, validation, test)


class ToyAttentionRegressor(nn.Module):
    """Predict a scalar from a six-value sequence using custom self-attention.

    A learned embedding is added at every sequence position. It is intentionally
    small: it gives the model a way to distinguish position 1 from position 4
    before the custom attention layer processes the sequence.
    """

    def __init__(
        self,
        sequence_length: int = 6,
        embedding_dim: int = 8,
        d_k: int = 8,
        d_v: int = 8,
        scale_scores: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(1, embedding_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(sequence_length, embedding_dim))
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
        self.attention = ScaledDotProductSelfAttention(embedding_dim, d_k, d_v, scale_scores)
        self.prediction_head = nn.Linear(d_v, 1)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scalar predictions ``[batch]`` and attention weights."""
        if X.ndim != 2 or X.shape[1] != self.sequence_length:
            raise ValueError(f"X must have shape [batch_size, {self.sequence_length}]")
        representations = self.input_projection(X.unsqueeze(-1)) + self.position_embeddings
        attended_values, attention_weights = self.attention(representations)
        sequence_representation = attended_values.mean(dim=1)
        predictions = self.prediction_head(sequence_representation).squeeze(-1)
        return predictions, attention_weights

    def forward_with_attention_details(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return predictions, weights, and underlying attention tensors for analysis."""
        if X.ndim != 2 or X.shape[1] != self.sequence_length:
            raise ValueError(f"X must have shape [batch_size, {self.sequence_length}]")
        representations = self.input_projection(X.unsqueeze(-1)) + self.position_embeddings
        attended_values, attention_weights, details = self.attention.forward_with_details(representations)
        predictions = self.prediction_head(attended_values.mean(dim=1)).squeeze(-1)
        return predictions, attention_weights, details


def train_toy_model(
    task_config: ToyTaskConfig | None = None,
    training_config: ToyTrainingConfig | None = None,
) -> ToyTrainingResult:
    """Train the toy regressor on CPU and return losses, metrics, and the model."""
    task_config = task_config or ToyTaskConfig()
    training_config = training_config or ToyTrainingConfig()
    datasets = make_toy_datasets(task_config)
    torch.manual_seed(training_config.model_seed)
    model = ToyAttentionRegressor(
        task_config.sequence_length,
        training_config.embedding_dim,
        training_config.d_k,
        training_config.d_v,
        training_config.scale_scores,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    loss_function = nn.MSELoss()
    loader_generator = torch.Generator().manual_seed(training_config.model_seed)
    train_loader = DataLoader(
        datasets.train, batch_size=training_config.batch_size, shuffle=True, generator=loader_generator
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []
    gradients_finite = True
    gradient_norm_total = 0.0
    gradient_steps = 0
    for _ in range(training_config.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions, _ = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            gradients_finite = gradients_finite and all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
            gradient_norm_total += sum(
                parameter.grad.square().sum().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            ) ** 0.5
            gradient_steps += 1
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)
        training_losses.append(total_loss / total_samples)
        validation_losses.append(_evaluate_mse(model, datasets.validation))

    test_mse = _evaluate_mse(model, datasets.test)
    with torch.no_grad():
        test_predictions, _ = model(datasets.test.tensors[0])
        test_mae = (test_predictions - datasets.test.tensors[1]).abs().mean().item()
    return ToyTrainingResult(
        model,
        training_losses,
        validation_losses,
        test_mse,
        test_mae,
        gradients_finite,
        gradient_norm_total / gradient_steps,
    )


def _evaluate_mse(model: ToyAttentionRegressor, dataset: TensorDataset) -> float:
    model.eval()
    with torch.no_grad():
        predictions, _ = model(dataset.tensors[0])
        return nn.functional.mse_loss(predictions, dataset.tensors[1]).item()
