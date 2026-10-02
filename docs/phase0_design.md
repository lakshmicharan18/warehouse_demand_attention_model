# Phase 0 Design

## Problem formulation

Predict warehouse demand for the next hour from the previous 24 hourly demand observations.

- Input: `x_t = [d_(t-23), ..., d_t]`, a 24-hour demand window.
- Output: `y_t = d_(t+1)`, the next hour's demand.

Demand is a continuous, non-negative synthetic quantity (for example, expected orders or picking workload per hour). The goal is to model realistic temporal structure, not to construct data that favors attention over a baseline.

## Synthetic warehouse environment

Hourly demand combines a stable base level with calendar effects, temporal dependence, random variation, and occasional operational events:

`d_t = max(0, base + daily(hour_t) + weekly(day_t) + temporal_t + event_t + noise_t)`

Where:

- `base` is the typical warehouse demand level.
- `daily(hour_t)` follows an operational day: lower overnight demand, increasing morning demand, moderate-to-high daytime demand, an evening peak, then declining nighttime demand.
- `weekly(day_t)` makes weekdays and weekends slightly different, such as lower or shifted weekend demand.
- `temporal_t` carries information from prior demand. Recent hours have the strongest influence, while selected earlier positions in the preceding day (for example, the same period earlier in the day) retain a smaller useful influence.
- `event_t` represents occasional promotions/busy periods (positive spikes) and operational slowdowns (negative drops).
- `noise_t` is moderate random variation.

The temporal component can be expressed simply as a weighted dependence on history:

`temporal_t = a1 * d_(t-1) + a2 * d_(t-2) + a24 * d_(t-24)`

`a1` is larger than `a2` and `a24`; all coefficients are selected so the process remains stable. This makes recent demand highly relevant while preserving useful daily-lag information within a 24-hour input window.

## Statistical and noise assumptions

- Demand is non-negative and hourly observations are ordered in time.
- Daily and weekly effects repeat approximately, but not perfectly.
- Noise is zero-mean, moderate, and independent between most hours; a normal or similarly light-tailed distribution is sufficient for the initial dataset.
- Events occur infrequently and are independent of normal noise. Spikes add a short-lived positive effect; drops add a short-lived negative effect, with demand clipped at zero if necessary.
- Event timing is not designed around a model or baseline. Some events become predictable after they begin, while isolated first-hour events may be inherently unpredictable.

## Expected correlations

- Adjacent hours are positively correlated, especially during sustained operating periods.
- Demand correlates with hour of day, including comparable periods on nearby days.
- Earlier informative positions, particularly the daily lag, have weaker but meaningful dependence.
- Weekday/weekend behavior creates correlation with day of week.
- Noise and isolated events prevent these correlations from becoming deterministic.

## Data splitting and evaluation

Use chronological splits only: an early segment for training, the following segment for validation, and the latest segment for test. Windows must not cross split boundaries in a way that exposes future targets during training or tuning.

Evaluate next-hour forecasts with:

- MAE: average absolute prediction error; easy to interpret in demand units.
- RMSE: penalizes larger forecasting errors, including missed spikes.

## Baselines

1. **Last-observation baseline:** `d_(t+1) = d_t`.
2. **Optional 24-hour moving-average baseline:** `d_(t+1)` equals the mean of the prior 24 observations.

## Hypotheses

1. The attention model may outperform the last-observation baseline because next-hour demand depends on multiple historical positions, not only the most recent hour.
2. Scaled attention should train more stably than unscaled attention because scaling limits excessively large attention logits and overly concentrated softmax outputs.

## Possible failure modes

- A sudden extreme demand spike may not be predictable from the previous 24 hours.
- Higher noise or changed demand patterns can cause distribution shift and reduce accuracy.
- Attention may emphasize historical positions that are not genuinely useful.
- Recurring daily patterns may be strong enough that a simple baseline remains competitive.

## Generalization experiment

Train under normal warehouse conditions. Then evaluate on a separate, chronologically later changed distribution with higher noise and larger or more frequent demand spikes. Compare MAE and RMSE against normal-condition test performance and the baselines to assess robustness.
