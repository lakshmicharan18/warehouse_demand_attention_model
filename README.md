# Warehouse Demand Attention Model

Build scaled dot-product self-attention from first principles in PyTorch, then use it to predict next-hour synthetic warehouse demand from the previous 24 hours.

## Workflow

Phase 0 design → synthetic data → first-principles attention → gradient verification → toy task → scaled/unscaled ablation → warehouse forecasting → baselines → distribution shift → failure analysis.

Two isolated mathematical bonus experiments extend this workflow: manual backpropagation through attention and an empirical `d_k` scaling study.

## Main results

| Method | Normal MAE | Normal RMSE |
|---|---:|---:|
| Attention model | 8.40 | 11.69 |
| Last observation | 12.97 | 16.71 |
| 24-hour moving average | 30.99 | 35.19 |

Under the configured shifted distribution, attention degraded to MAE 12.83 / RMSE 18.33 but remained ahead of the two baselines. Details are in [docs/final_report.md](docs/final_report.md).

## Mathematical validation and bonus experiments

- **Finite differences:** central-difference gradients agreed with PyTorch autograd, with maximum absolute difference `5.62e-11`.
- **Manual backpropagation:** explicit chain-rule gradients for `W_Q`, `W_K`, `W_V`, and `X` agreed with autograd to float64 precision (largest absolute difference `3.47e-18`). See [manual derivation](docs/manual_backpropagation.md).
- **Why scale by √dₖ:** across `d_k = 4…128`, raw logit standard deviation increased from 2.00 to 11.30, while scaled standard deviation stayed near 1.00. Unscaled attention became increasingly concentrated; scaled entropy remained stable near 2.74. See the [dimension scaling experiment](docs/dimension_scaling_experiment.md).

## Structure

- `src/`: data, custom attention, models, and experiment utilities.
- `experiments/`: reproducible command-line experiments.
- `tests/`: focused correctness tests.
- `docs/`: design, evidence, and submission documentation.
- `outputs/`: generated CSVs and plots.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Reproducibility Environment

Verified with Python 3.13.11, PyTorch 2.12.1+cu130, NumPy 2.4.2, matplotlib 3.10.8, and pytest 9.1.1. The exact compatible package requirements are pinned in `requirements.txt`; experiment seeds and configurations are defined in the corresponding source dataclasses.

## Reproduce experiments

Run each command from the repository root:

```bash
python experiments/generate_dataset.py
python experiments/inspect_attention.py
python experiments/verify_gradients.py
python experiments/manual_backprop_experiment.py
python experiments/dimension_scaling_experiment.py
python experiments/train_toy_task.py
python experiments/attention_ablation.py
python experiments/train_warehouse_model.py
python experiments/generalization_experiment.py
python experiments/failure_analysis.py
```

## Limitations

This is a synthetic-data study with a small single-head attention model and point forecasts. Rare external events can be unpredictable without event features; results do not establish real-warehouse performance or causal meaning for attention weights.

## Out of scope / not required

Full Transformer architectures, multi-head attention, production deployment, frontend/API/database work, large datasets, distributed training, and GPU optimization are not required for this assignment and were intentionally not implemented.

## Implemented bonus work

- Manual backpropagation through the attention block: [documentation](docs/manual_backpropagation.md) and `python experiments/manual_backprop_experiment.py`.
- `d_k` scaling study: [documentation](docs/dimension_scaling_experiment.md) and `python experiments/dimension_scaling_experiment.py`.

## Optional / bonus work not implemented

Paper reproduction, LoRA-style adaptation, a JAX implementation, and numerical or memory optimization remain bonus extensions, not assignment requirements.
