"""Find and visualize the largest shifted-test forecast failure."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from warehouse_demand_attention.failure_analysis import analyze_shifted_failure, error_percentiles  # noqa: E402
from warehouse_demand_attention.training import train_warehouse_model  # noqa: E402

def main() -> None:
    analysis = analyze_shifted_failure(train_warehouse_model())
    case, stats = analysis.case, error_percentiles(analysis.absolute_errors)
    print(f"Selected test index: {case.test_index}")
    print(f"Actual={case.actual:.3f}; prediction={case.prediction:.3f}; absolute_error={case.absolute_error:.3f}; event={case.target_event}")
    print(f"Last={case.last_prediction:.3f}; moving_average={case.moving_average_prediction:.3f}")
    print("Error percentiles:", stats)
    print(f"Spike-target MAE={analysis.spike_target_errors.mean():.3f}; non-spike-target MAE={analysis.non_spike_target_errors.mean():.3f}")
    top = np.argsort(case.attention_weights)[-3:][::-1]
    print("Top attention:", [(int(24-i), round(float(case.attention_weights[i]), 3)) for i in top])
    output=PROJECT_ROOT/"outputs"
    with (output/"top_failure_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["test_index","actual","prediction","absolute_error"])
        for i in np.argsort(analysis.absolute_errors)[-5:][::-1]: w.writerow([i, analysis.case.actual if i==case.test_index else "", analysis.case.prediction if i==case.test_index else "", analysis.absolute_errors[i]])
    x=np.arange(-24,0); plt.figure(figsize=(9,4)); plt.plot(x,case.history,marker="o",label="History"); plt.scatter([1],[case.actual],label="Actual next hour",zorder=3); plt.scatter([1],[case.prediction],label="Attention prediction",zorder=3); plt.scatter([1],[case.last_prediction],label="Last observation",zorder=3); plt.scatter([1],[case.moving_average_prediction],label="Moving average",zorder=3); plt.xlabel("Hours relative to target"); plt.ylabel("Demand"); plt.title("Largest Shifted-Test Forecast Error"); plt.legend(); plt.tight_layout(); plt.savefig(output/"failure_case.png",dpi=150); plt.close()

if __name__ == "__main__": main()
