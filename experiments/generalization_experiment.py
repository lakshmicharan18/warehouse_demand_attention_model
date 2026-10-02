"""Train normally once, then evaluate the frozen model under demand shift."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.generalization import evaluate_frozen_model  # noqa: E402
from warehouse_demand_attention.training import train_warehouse_model  # noqa: E402


def main() -> None:
    result = evaluate_frozen_model(train_warehouse_model())
    print("Method | Normal MAE | Shifted MAE | MAE change | Normal RMSE | Shifted RMSE | RMSE change")
    rows = []
    for method in result.normal:
        normal, shifted = result.normal[method], result.shifted[method]
        rows.append((method, normal.mae, shifted.mae, shifted.mae-normal.mae, normal.rmse, shifted.rmse, shifted.rmse-normal.rmse))
        print(f"{method}: {normal.mae:.3f}, {shifted.mae:.3f}, {shifted.mae-normal.mae:+.3f}; {normal.rmse:.3f}, {shifted.rmse:.3f}, {shifted.rmse-normal.rmse:+.3f}")
    for label, stats, config in (("Normal", result.normal_statistics, result.normal_config), ("Shifted", result.shifted_statistics, result.shifted_config)):
        print(f"{label} data: mean={stats.mean:.2f}, std={stats.std:.2f}, min={stats.minimum:.2f}, max={stats.maximum:.2f}, spikes={stats.spike_count}, drops={stats.drop_count}; noise={config.noise_std}, spike_p={config.spike_probability}, spike_range={config.spike_magnitude_range}")
    for example in range(3):
        top = np.argsort(result.shifted_attention_weights[example])[-3:][::-1]
        print(f"Shifted example {example} top positions: " + ", ".join(f"{24-index}h ago ({result.shifted_attention_weights[example,index]:.3f})" for index in top))
    output = PROJECT_ROOT / "outputs"
    with (output / "generalization_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["method", "normal_mae", "shifted_mae", "mae_change", "normal_rmse", "shifted_rmse", "rmse_change"]); writer.writerows(rows)
    length = 168
    plt.figure(figsize=(9, 4)); plt.plot(result.shifted_targets[:length], label="Actual"); plt.plot(result.shifted_attention_predictions[:length], label="Attention model"); plt.plot(result.shifted_last_predictions[:length], label="Last observation", alpha=0.7); plt.xlabel("Shifted test-set hour"); plt.ylabel("Demand"); plt.title("Frozen Model on Shifted Warehouse Demand"); plt.legend(); plt.tight_layout(); plt.savefig(output / "generalization_predictions.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
