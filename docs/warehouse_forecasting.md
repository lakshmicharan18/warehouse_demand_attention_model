# Warehouse Forecasting

## Problem

Use the previous 24 hourly demand observations to forecast next-hour demand.

## Model

The model maps `24 hours -> scalar projection + learned relative-position embeddings -> custom scaled self-attention -> final-position context -> linear prediction`. The final position represents the most recent hour and can attend to every historical hour. A causal mask is unnecessary because the forecast target is outside the input window.

## Baselines

The last-observation baseline forecasts the most recent demand. The moving-average baseline forecasts the mean of all 24 input hours.

## Data split

Supervised windows are split chronologically: 3,007 train, 644 validation, and 645 test samples. This prevents future periods from being used for model selection.

## Normalization

Demand mean and standard deviation are fitted only on training input windows. The same statistics normalize train, validation, and test inputs/targets; predictions are converted back to demand units before MAE and RMSE are calculated.

## Evaluation

MAE gives average error in demand units; RMSE emphasizes large errors. On the normal synthetic test set:

| Method | MAE | RMSE |
|---|---:|---:|
| Last observation | 12.97 | 16.71 |
| 24-hour moving average | 30.99 | 35.19 |
| Attention model | 8.40 | 11.69 |

## Attention observation

For the first inspected test windows, final-position attention emphasized approximately 23 hours ago, 19 hours ago, and either 24 or 2 hours ago. These weights are behavioral evidence from this fitted model, not proof of causal importance.
