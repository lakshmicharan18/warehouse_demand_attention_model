# Gradient Verification

The attention module's autograd gradients are checked independently for selected entries of `W_Q`, `W_K`, and `W_V`. The scalar loss is the mean squared attention output.

For each selected parameter value `w`, central finite differences estimate the derivative:

`(L(w + epsilon) - L(w - epsilon)) / (2 * epsilon)`

The verification uses a deterministic `[1, 3, 2]` input, `d_k = 2`, `d_v = 2`, `torch.float64`, and `epsilon = 1e-6`. In the fixed verification example, the maximum observed absolute difference is `5.62e-11` and maximum relative difference is `1.31e-08`; the check passes the configured tolerances. Run `python experiments/verify_gradients.py` to reproduce the result.
