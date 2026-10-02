"""Run and display finite-difference verification for self-attention gradients."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from warehouse_demand_attention.gradient_verification import verify_gradients  # noqa: E402


def main() -> None:
    try:
        report = verify_gradients()
    except AssertionError as error:
        print(f"Gradient verification: FAILED ({error})")
        raise SystemExit(1) from error

    print("Parameter | Index  | Autograd       | Numerical      | Abs Diff       | Rel Diff")
    for result in report.results:
        print(
            f"{result.parameter_name:<9} | {str(result.index):<6} | "
            f"{result.autograd_gradient: .8e} | {result.numerical_gradient: .8e} | "
            f"{result.absolute_difference:.2e} | {result.relative_difference:.2e}"
        )
    print(f"Epsilon: {report.epsilon:.1e}")
    print(f"Maximum absolute difference: {report.maximum_absolute_difference:.2e}")
    print(f"Maximum relative difference: {report.maximum_relative_difference:.2e}")
    print("Gradient verification: PASSED")


if __name__ == "__main__":
    main()
