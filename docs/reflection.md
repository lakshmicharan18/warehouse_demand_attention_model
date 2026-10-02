# Reflection

1. I expected multiple historical lags to help next-hour forecasting and scaled attention to be more stable.
2. Multiple-lag attention beat both baselines on normal synthetic demand; scaling produced the predicted logit/entropy differences.
3. Scaling was only partially supported: the toy task was simple enough that unscaled attention trained well too.
4. It was surprising that clear softmax/logit differences produced almost identical toy-task accuracy, while shifted spikes caused large forecasting degradation.
5. The hardest technical problem was independently verifying autograd with carefully restored float64 central perturbations.
6. The selected failure was the largest shifted spike: 227.19 actual versus 125.92 prediction.
7. I understand better how scaling affects logits without necessarily changing performance on an easy task, and how event absence limits point forecasts.
8. It remains uncertain how these results transfer to real operations or richer event signals.
9. Codex assisted with implementation scaffolding, tests, analyses, and documentation.
10. I verified suggestions through tests, numerical gradient checks, plotted outputs, and fixed-seed reruns rather than accepting results unexamined.
11. With one more day, I would examine event features and calibrated probabilistic forecasts without changing the core attention comparison.
