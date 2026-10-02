# AI Assistance Log

| Milestone | Tool | Task requested | Assistance received | Candidate review/modification | Verification performed | Resulting understanding |
|---|---|---|---|---|---|---|
| Project planning | Codex | Create a minimal ML project. | Proposed project layout. | Scope kept to local experimentation. | File-layout review. | The work could remain infrastructure-light. |
| Phase 0 design | Codex | Define realistic demand design. | Drafted equations and hypotheses. | Checked against later generator. | Document review. | Daily, weekly, lag, and event effects were sufficient. |
| Synthetic data generator | Codex | Implement configured synthetic data. | Generated candidate code/tests. | Preserved explicit seed and event logic. | Fixed-seed tests and plots. | Temporal structure is visible but non-deterministic. |
| Self-attention implementation | Codex | Implement raw-tensor attention. | Drafted Q/K/V implementation. | Kept operations explicit; excluded high-level APIs. | Shape, softmax, and gradient tests. | Attention mechanics are inspectable. |
| Gradient verification | Codex | Check autograd numerically. | Added central-difference candidate. | Used float64 and parameter restoration. | Numerical/autograd comparison. | The custom operation is gradient-consistent. |
| Toy learning task | Codex | Demonstrate learning. | Drafted small positional toy model. | Kept sequence task controlled. | Loss, metric, and attention inspection. | Attention can train on position-specific signal. |
| Scaled/unscaled ablation | Codex | Compare score scaling. | Added matched-run utilities. | Held data, seeds, and architecture fixed. | Three-seed metrics and loss plot. | Scaling changed logits/entropy more than easy-task accuracy. |
| Warehouse forecasting | Codex | Train next-hour forecaster and baselines. | Drafted model/training evaluation. | Used chronological split and train-only normalization. | Metrics, plots, and tests. | Attention beat these synthetic baselines. |
| Distribution shift | Codex | Evaluate frozen model under shift. | Added shifted evaluator. | Reused normal normalization and frozen weights. | Shift metrics and parameter test. | More/noisier spikes degraded all methods. |
| Failure analysis | Codex | Analyze genuine maximum error. | Added reproducible selector/plot. | Selected by absolute error, not appearance. | Event metadata, percentiles, baseline comparison. | Rare spikes are a major limitation. |
| Final documentation/audit | Codex | Prepare submission artifacts. | Consolidated reports and audit suggestions. | Cross-checked reported values with outputs. | Tests, command runs, documentation review. | Claims are bounded by synthetic evidence. |

AI assistance was used transparently for candidate implementation and documentation. Tests, numerical checks, controlled experiments, result inspection, and documentation review were used to verify the work rather than treating generated output as unquestioned.
