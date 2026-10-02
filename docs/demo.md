# Technical Review Demo (20–30 minutes)

Run commands from the repository root.

## 1. Dataset

`python experiments/generate_dataset.py`

Observe 4,320 hourly observations, event counts, sequence shapes, and `outputs/warehouse_demand_first_7_days.png`. Explain that daily/weekly effects, lags, noise, and events make the forecasting problem nontrivial.

## 2. Attention

Open `src/warehouse_demand_attention/attention.py`, then run `python experiments/inspect_attention.py`.

Trace `X → Q/K/V → scores → scaling → stable softmax → output`; observe compatible shapes and attention rows summing to one.

## 3. Gradient verification

`python experiments/verify_gradients.py`

Observe tiny central-difference versus autograd differences. Explain this verifies PyTorch's gradients through the custom attention operations.

## 4. Toy task

`python experiments/train_toy_task.py`

Observe decreasing loss and attention focused on the target-relevant positions; this demonstrates a real learning loop, not only a forward pass.

## 5. Ablation

`python experiments/attention_ablation.py`

Observe larger, more concentrated unscaled logits but similar task accuracy. Explain the hypothesis was partially supported.

## 6. Warehouse forecasting

`python experiments/train_warehouse_model.py`

Observe attention versus baseline MAE/RMSE and `outputs/warehouse_predictions.png`. Explain train-only normalization and chronological splitting.

## 7. Generalization

`python experiments/generalization_experiment.py`

Observe all methods degrade under higher noise and stronger/more frequent spikes, while attention remains best among these methods.

## 8. Failure

`python experiments/failure_analysis.py`

Show `outputs/failure_case.png`. Explain why an explicit, unseen large spike cannot be reliably inferred from demand history alone.
