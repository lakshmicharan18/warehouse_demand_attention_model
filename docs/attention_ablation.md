# Scaled vs Unscaled Attention Ablation

## Hypothesis

Scaled attention should train more stably than unscaled attention because dividing by `sqrt(d_k)` limits unnecessarily large logits and overly concentrated softmax weights.

## Setup

The existing six-position toy regression task was run with three initialization seeds (`42`, `123`, `2026`). Both variants used the same fixed datasets (2,000/400/400), architecture, `d_k = 32`, optimizer (Adam), learning rate (`1e-3`), batch size (`128`), and 50 epochs. The only intended difference was whether scores were divided by `sqrt(d_k)`.

## Results

Means across three seeds:

| Metric | Scaled | Unscaled |
|---|---:|---:|
| Final training loss | 0.000139 | 0.000172 |
| Validation loss | 0.000138 | 0.000135 |
| Test MSE | 0.000125 | 0.000128 |
| Test MAE | 0.008876 | 0.008830 |
| Mean maximum attention weight | 0.683657 | 0.691761 |
| Attention entropy | 0.735856 | 0.683917 |
| Mean absolute logit | 3.234437 | 3.711866 |
| Maximum absolute logit | 4.714330 | 7.386369 |
| Mean gradient norm | 0.088690 | 0.118865 |

All runs had finite gradients.

## Observation

Unscaled attention had larger logits, slightly more concentrated weights, lower entropy, and larger average gradient norms. Both variants nonetheless trained smoothly and reached nearly identical test errors on this simple task.

## Comparison with hypothesis

The evidence partially supports the hypothesis: the expected logit and concentration differences appeared, but neither variant became unstable and scaling did not produce a meaningful prediction advantage here.

## Mathematical explanation

Dot products grow with representation dimension. Dividing by `sqrt(d_k)` reduced the softmax logits in this experiment, producing less concentrated distributions. The unscaled variant's larger logits corresponded to higher gradient norms, but the task was simple enough that useful gradient flow and learning remained intact in both conditions.
