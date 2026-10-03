"""Display manual attention backward gradients against PyTorch autograd."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from warehouse_demand_attention.manual_backprop import compare_manual_to_autograd  # noqa: E402

def main() -> None:
    result = compare_manual_to_autograd()
    print("Gradient | Manual max | Autograd max | Max abs diff | Max rel diff")
    for name in ("dW_Q", "dW_K", "dW_V", "dX"):
        print(f"{name:<8} | {result.manual[name].abs().max():.8e} | {result.autograd[name].abs().max():.8e} | {result.max_absolute_differences[name]:.2e} | {result.max_relative_differences[name]:.2e}")
    tolerance = 1e-10
    if all(value <= tolerance for value in result.max_absolute_differences.values()):
        print("Manual backpropagation verification: PASSED")
    else:
        print("Manual backpropagation verification: FAILED")
        raise SystemExit(1)

if __name__ == "__main__": main()
