"""Train and evaluate the custom-attention warehouse next-hour forecaster."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.baselines import last_observation_baseline, moving_average_baseline  # noqa: E402
from warehouse_demand_attention.training import mae_rmse, train_warehouse_model  # noqa: E402


def main() -> None:
    result = train_warehouse_model()
    raw_test_X = result.data.normalizer.inverse_transform(result.data.test_X)
    target = result.data.test_y
    last = last_observation_baseline(raw_test_X)
    moving = moving_average_baseline(raw_test_X)
    rows = [("Last observation", *mae_rmse(last, target)), ("24-hour moving average", *mae_rmse(moving, target)), ("Attention model", *mae_rmse(result.test_predictions, target))]
    print(f"Final train loss: {result.training_losses[-1]:.6f}")
    print(f"Best validation loss: {result.validation_losses[result.best_epoch - 1]:.6f} (epoch {result.best_epoch})")
    print("Method                  MAE       RMSE")
    for method, mae, rmse in rows:
        print(f"{method:<23} {mae:.4f}    {rmse:.4f}")
    for example in range(3):
        top = np.argsort(result.final_attention_weights[example])[-3:][::-1]
        description = ", ".join(f"{24-index}h ago ({result.final_attention_weights[example, index]:.3f})" for index in top)
        print(f"Test example {example} strongest final-position attention: {description}")

    output = PROJECT_ROOT / "outputs"
    with (output / "warehouse_model_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["method", "mae", "rmse"]); writer.writerows(rows)
    epochs = range(1, len(result.training_losses) + 1)
    plt.figure(figsize=(7, 4)); plt.plot(epochs, result.training_losses, label="Training"); plt.plot(epochs, result.validation_losses, label="Validation"); plt.xlabel("Epoch"); plt.ylabel("Normalized MSE"); plt.title("Warehouse Attention Model Training Loss"); plt.legend(); plt.tight_layout(); plt.savefig(output / "warehouse_training_loss.png", dpi=150); plt.close()
    length = 168
    plt.figure(figsize=(9, 4)); plt.plot(target[:length], label="Actual"); plt.plot(result.test_predictions[:length], label="Attention model"); plt.plot(last[:length], label="Last observation", alpha=0.7); plt.xlabel("Test-set hour"); plt.ylabel("Demand"); plt.title("Warehouse Demand Predictions (Test Section)"); plt.legend(); plt.tight_layout(); plt.savefig(output / "warehouse_predictions.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
