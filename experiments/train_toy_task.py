"""Train and inspect the small attention-based sequence regression task."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.toy_task import (  # noqa: E402
    ToyTaskConfig,
    make_toy_datasets,
    train_toy_model,
)


def main() -> None:
    task_config = ToyTaskConfig()
    result = train_toy_model(task_config)
    datasets = make_toy_datasets(task_config)

    print(f"Initial training loss: {result.training_losses[0]:.6f}")
    print(f"Final training loss: {result.training_losses[-1]:.6f}")
    print(f"Validation loss: {result.validation_losses[-1]:.6f}")
    print(f"Test MSE: {result.test_mse:.6f}")
    print(f"Test MAE: {result.test_mae:.6f}")
    print(f"Finite gradients during training: {result.gradients_finite}")

    test_X, test_y = datasets.test.tensors
    with torch.no_grad():
        predictions, weights = result.model(test_X[:3])
    print("Example predictions vs targets:")
    for prediction, target in zip(predictions, test_y[:3]):
        print(f"prediction={prediction.item():.4f}, target={target.item():.4f}")
    print("First sequence attention weights:")
    print(weights[0])

    plot_path = PROJECT_ROOT / "outputs" / "toy_training_loss.png"
    epochs = range(1, len(result.training_losses) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, result.training_losses, label="Training loss")
    plt.plot(epochs, result.validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Toy Attention Task Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training-loss plot: {plot_path}")


if __name__ == "__main__":
    main()
