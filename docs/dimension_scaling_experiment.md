# Dimension Scaling Experiment

## Question

Why divide attention dot-product scores by `sqrt(d_k)`?

## Pre-experiment hypothesis

As `d_k` increases, unscaled attention logits should grow in magnitude and attention should become more concentrated. Dividing by `sqrt(d_k)` should keep the scaled logit distribution substantially more stable across dimensions and reduce the tendency toward softmax concentration.

## Mathematical intuition

For approximately independent zero-mean, unit-variance components, `Q·K = sum(q_i k_i)` adds `d_k` terms. Thus `Var(Q·K)` grows approximately with `d_k`, while standard deviation grows with `sqrt(d_k)`. Scaling is expected to counter that growth; this is an approximation, not a claim about every trained representation.

## Setup

Tested `d_k = 4, 8, 16, 32, 64, 128` with five fixed seeds (`11, 22, 33, 44, 55`), float64 Q/K/V sampled from Normal(0,1), batch size 128, and sequence length 24. The objective for the gradient measurement was `mean((softmax(scores) @ V)^2)`; reported values are trial means.

## Results

| d_k | Raw std | Scaled std | Raw std / sqrt(d_k) | Unscaled entropy | Scaled entropy | Unscaled max weight | Scaled max weight | Unscaled grad norm | Scaled grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.0014 | 1.0007 | 1.0007 | 2.0631 | 2.7625 | 0.3740 | 0.1858 | 8.430e-03 | 2.497e-03 |
| 8 | 2.8377 | 1.0033 | 1.0033 | 1.5133 | 2.7487 | 0.5223 | 0.1917 | 1.159e-02 | 1.960e-03 |
| 16 | 3.9893 | 0.9973 | 0.9973 | 1.0481 | 2.7471 | 0.6484 | 0.1918 | 1.566e-02 | 1.584e-03 |
| 32 | 5.6513 | 0.9990 | 0.9990 | 0.6829 | 2.7419 | 0.7576 | 0.1949 | 1.933e-02 | 1.417e-03 |
| 64 | 7.9752 | 0.9969 | 0.9969 | 0.4705 | 2.7418 | 0.8241 | 0.1939 | 2.425e-02 | 1.295e-03 |
| 128 | 11.3044 | 0.9992 | 0.9992 | 0.3179 | 2.7403 | 0.8768 | 0.1938 | 2.964e-02 | 1.215e-03 |

## Interpretation

Raw logit standard deviation grew approximately with `sqrt(d_k)`, and `raw_std / sqrt(d_k)` stayed near one. Scaling held the logit standard deviation near one across dimensions. Unscaled softmax became much more concentrated (falling entropy and rising maximum weight); scaled attention stayed nearly constant. The unscaled query-gradient norm increased here, while the scaled norm declined modestly. These norms are observations for this objective, not a claim that either magnitude is automatically better.

## Hypothesis conclusion

**Supported.** The experiment showed the expected raw-score growth, scaled-score stabilization, and unscaled concentration across dimensions.
