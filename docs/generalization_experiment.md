# Distribution-Shift Generalization Experiment

## Original distribution

The normal synthetic warehouse data uses noise standard deviation `8.0`, spike probability `0.015`, and spike range `(30.0, 65.0)`. Its mean demand is `107.60`, standard deviation `35.51`, range `9.12` to `221.98`, with 57 spikes and 50 drops.

## Modified distribution

The shifted data keeps the same daily/weekly/lag process and 24-hour windows, but uses noise standard deviation `12.0`, spike probability `0.03`, and spike range `(37.5, 97.5)`. It has mean `109.34`, standard deviation `38.24`, range `0.00` to `283.02`, with 114 spikes and 50 drops.

## Expected impact

We expect performance to degrade under distribution shift because the shifted data contains more noise and stronger demand spikes than the model observed during training. However, if the model learned useful temporal structure such as daily dependencies, it should still retain some predictive ability.

## Results

The attention model was trained only on normal data and evaluated with its original training-set normalization statistics.

| Method | Normal MAE | Shifted MAE | Normal RMSE | Shifted RMSE |
|---|---:|---:|---:|---:|
| Last observation | 12.97 | 18.33 | 16.71 | 24.33 |
| 24-hour moving average | 30.99 | 33.19 | 35.19 | 39.06 |
| Attention model | 8.40 | 12.83 | 11.69 | 18.33 |

## Observed impact

Attention MAE rose by `4.44` (52.9%) and RMSE by `6.64` (56.8%). Last observation also degraded substantially (41.3% MAE; 45.6% RMSE); moving average degraded less proportionally but remained much less accurate. Attention retained the best shifted performance, while larger spikes and noise increased large-error sensitivity most clearly in RMSE.

## Interpretation

The result matches the expectation: learned temporal structure remains useful, but it does not fully transfer to noisier, more event-heavy demand. Shifted final-position attention weights remain inspectable behavioral signals, not causal explanations.
