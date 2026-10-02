"""Generate and summarize the normal synthetic warehouse-demand dataset."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.data_generator import (  # noqa: E402
    create_supervised_sequences,
    chronological_split,
    generate_warehouse_demand,
)


def main() -> None:
    dataset = generate_warehouse_demand()
    X, y = create_supervised_sequences(dataset.demand)
    splits = chronological_split(X, y)

    print(f"Observations: {len(dataset.demand)}")
    print(f"Demand min: {dataset.demand.min():.2f}")
    print(f"Demand max: {dataset.demand.max():.2f}")
    print(f"Demand mean: {dataset.demand.mean():.2f}")
    print(f"Demand std: {dataset.demand.std():.2f}")
    print(f"Spike events: {dataset.spike_events.sum()}")
    print(f"Drop events: {dataset.drop_events.sum()}")
    print(f"X shape: {X.shape}; y shape: {y.shape}")
    print(
        "Split sizes: "
        f"train={len(splits.y_train)}, validation={len(splits.y_validation)}, test={len(splits.y_test)}"
    )

    output_path = PROJECT_ROOT / "outputs" / "warehouse_demand_sample.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "demand", "hour", "day_of_week", "spike_event", "drop_event"])
        for index in range(min(24 * 14, len(dataset.demand))):
            writer.writerow([
                dataset.timestamps[index].isoformat(), f"{dataset.demand[index]:.4f}",
                dataset.hours[index], dataset.day_of_week[index],
                bool(dataset.spike_events[index]), bool(dataset.drop_events[index]),
            ])
    print(f"Saved sample CSV: {output_path}")

    figure_path = PROJECT_ROOT / "outputs" / "warehouse_demand_first_7_days.png"
    hours_to_plot = min(24 * 7, len(dataset.demand))
    plt.figure(figsize=(10, 4))
    plt.plot(dataset.timestamps[:hours_to_plot], dataset.demand[:hours_to_plot])
    plt.title("Synthetic Warehouse Demand: First 7 Days")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    print(f"Saved first-7-days plot: {figure_path}")


if __name__ == "__main__":
    main()
