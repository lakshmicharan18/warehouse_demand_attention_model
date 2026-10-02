# Final Report

## 1. Problem

Predict next-hour warehouse demand using the previous 24 hourly observations.

## 2. Phase 0 hypotheses

Attention may beat last observation by using multiple historical positions; scaling by `sqrt(d_k)` may stabilize softmax behavior.

## 3. Synthetic environment

Demand combines daily and weekly seasonality, `t-1`/`t-2`/`t-24` dependence, Gaussian noise, and occasional spikes/drops.

## 4–5. Attention and verification

The project explicitly implements Q/K/V projections, dot-product scores, scaling, stable softmax, and weighted aggregation—without high-level attention APIs. Shape, normalization, and finite-gradient tests pass; central finite differences matched autograd with maximum absolute difference `5.62e-11`.

## 6. Toy learning

For `0.7*x[1] + 0.3*x[4] + noise`, final training/validation loss was `0.000120/0.000124`; test MSE/MAE was `0.000116/0.008668`.

## 7. Ablation

Scaled/unscaled runs used identical toy setups. Unscaled logits were larger (mean/max `3.71/7.39` vs `3.23/4.71`) and entropy lower (`0.684` vs `0.736`), but test accuracy was nearly identical. The stability hypothesis was partially supported.

## 8. Warehouse prediction

| Method | MAE | RMSE |
|---|---:|---:|
| Attention | 8.40 | 11.69 |
| Last observation | 12.97 | 16.71 |
| Moving average | 30.99 | 35.19 |

## 9. Generalization

Shifted noise/spikes raised attention MAE/RMSE to `12.83/18.33`; it remained better than baselines but degraded materially.

## 10. Failure investigation

The largest shifted error was an explicit spike: actual `227.19`, prediction `125.92`, error `101.27`. Spike-target MAE (`63.90`) greatly exceeded non-spike MAE (`11.45`).

## 11. Evidence

Evidence supports that this custom attention is mathematically consistent, learns a controlled sequence task, beats these baselines on this synthetic task, and degrades under shift. It does not show universal attention superiority, real-warehouse generalization, or causal attention explanations.

## 12. Limitations

Synthetic data, a small single-head point forecaster, no external event features, and limited distribution-shift scenarios constrain conclusions.
