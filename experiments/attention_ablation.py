"""Compare matched scaled and unscaled custom-attention toy-task runs."""

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

from warehouse_demand_attention.ablation import (  # noqa: E402
    metric_mean_and_std,
    run_attention_ablation,
)


METRICS = (
    ("Initial train loss", "initial_training_loss"),
    ("Final train loss", "final_training_loss"),
    ("Validation loss", "validation_loss"),
    ("Test MSE", "test_mse"),
    ("Test MAE", "test_mae"),
    ("Mean max attention", "mean_max_attention_weight"),
    ("Attention entropy", "attention_entropy"),
    ("Mean |logit|", "mean_absolute_logit"),
    ("Max |logit|", "max_absolute_logit"),
    ("Gradient norm", "mean_gradient_norm"),
)


def main() -> None:
    summary = run_attention_ablation()
    print("Metric                 Scaled (mean±std)    Unscaled (mean±std)")
    for label, field in METRICS:
        scaled_mean, scaled_std = metric_mean_and_std(summary.scaled_runs, field)
        unscaled_mean, unscaled_std = metric_mean_and_std(summary.unscaled_runs, field)
        print(f"{label:<22} {scaled_mean:.6f}±{scaled_std:.6f}    {unscaled_mean:.6f}±{unscaled_std:.6f}")
    print(f"Finite gradients:      {all(run.gradients_finite for run in summary.scaled_runs)}                 {all(run.gradients_finite for run in summary.unscaled_runs)}")

    output_dir = PROJECT_ROOT / "outputs"
    _save_summary_csv(summary, output_dir / "attention_ablation_summary.csv")
    _save_loss_plot(summary, output_dir / "attention_ablation_loss.png")
    print(f"Saved summary CSV: {output_dir / 'attention_ablation_summary.csv'}")
    print(f"Saved loss plot: {output_dir / 'attention_ablation_loss.png'}")


def _save_summary_csv(summary, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["variant", "seed", *[field for _, field in METRICS], "gradients_finite"])
        for variant, runs in (("scaled", summary.scaled_runs), ("unscaled", summary.unscaled_runs)):
            for run in runs:
                writer.writerow([variant, run.seed, *[getattr(run, field) for _, field in METRICS], run.gradients_finite])


def _save_loss_plot(summary, path: Path) -> None:
    scaled_losses = np.array([run.training_losses for run in summary.scaled_runs])
    unscaled_losses = np.array([run.training_losses for run in summary.unscaled_runs])
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, scaled_losses.shape[1] + 1), scaled_losses.mean(axis=0), label="Scaled")
    plt.plot(np.arange(1, unscaled_losses.shape[1] + 1), unscaled_losses.mean(axis=0), label="Unscaled")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Scaled vs Unscaled Attention: Toy Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
