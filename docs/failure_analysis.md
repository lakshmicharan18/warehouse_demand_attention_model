# Failure Analysis

## Scenario

The reproducible maximum-absolute-error shifted test example is index `203`. Its target is an explicit spike event. The preceding 24-hour window contains two earlier spikes and no drops.

## Expected behavior

With only historical demand, the reasonable expectation is that the model follows learned daily and recent temporal structure. It cannot be expected to predict a future event whose trigger is absent from its inputs.

## Actual behavior

Actual next-hour demand was `227.19`; the attention prediction was `125.92`, an absolute error of `101.27` (44.6%). Last observation predicted `127.51`; the moving average predicted `101.10`.

## Evidence

History, from 24 hours ago to 1 hour ago:

`108.26, 121.15, 104.88, 133.94, 123.80, 131.04, 110.75, 160.26, 76.00, 81.09, 94.45, 78.01, 50.20, 37.91, 43.65, 58.16, 71.09, 70.95, 89.80, 100.18, 118.78, 153.19, 181.33, 127.51`.

The final-position attention weights were largest at 7 hours ago (`0.272`), 10 hours ago (`0.214`), and 8 hours ago (`0.176`). Across shifted tests, median absolute error was `8.74`, 90th percentile `26.31`, 95th percentile `35.71`, and maximum `101.27`. Spike-target MAE was `63.90`, versus `11.45` for non-spike targets.

## Explanation

The evidence supports an unpredictable-event and distribution-shift failure: the target was a large explicit spike, while the model and both baselines regressed toward the normal historical range. Attention emphasized ordinary historical positions, which is behavioral evidence of its reliance on history rather than a causal explanation.

## Potential improvement

Potential remedies include promotion/operational-event indicators, training data with a wider spike range, calendar and operational features, or probabilistic forecasts that express event uncertainty. These are not implemented here.

## Limitation

Attention cannot reliably predict a spike when information needed to anticipate it is absent from the previous 24 hourly demand values.
