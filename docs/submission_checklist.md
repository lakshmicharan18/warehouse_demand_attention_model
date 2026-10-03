# Submission Checklist

| Requirement | Evidence/file | Status |
|---|---|---|
| Phase 0 design and hypotheses | `docs/phase0_design.md` | Complete |
| Mathematical formulation | `docs/attention_mathematics.md` | Complete |
| Synthetic dataset | `src/warehouse_demand_attention/data_generator.py` | Complete |
| First-principles attention | `src/warehouse_demand_attention/attention.py` | Complete |
| Gradient verification | `docs/gradient_verification.md` | Complete |
| Optional manual backpropagation | `docs/manual_backpropagation.md` | Complete (bonus) |
| Optional d_k scaling experiment | `docs/dimension_scaling_experiment.md` | Complete (bonus) |
| Toy task and training dynamics | `toy_task.py`, `outputs/toy_training_loss.png` | Complete |
| Scaled/unscaled ablation | `docs/attention_ablation.md` | Complete |
| Warehouse forecasting | `docs/warehouse_forecasting.md` | Complete |
| Baselines and evaluation | `baselines.py`, `warehouse_model_results.csv` | Complete |
| Distribution shift | `docs/generalization_experiment.md` | Complete |
| Failure analysis | `docs/failure_analysis.md` | Complete |
| Tests | `tests/` | Complete |
| Reproducible commands | `README.md`, `docs/demo.md` | Complete |
| Debugging process | `docs/debugging_evidence.md` | Complete |
| Tested Python/framework versions | `README.md`, `requirements.txt` | Complete |
| Reflection | `docs/reflection.md` | Complete |
| AI assistance log | `docs/ai_assistance_log.md` | Complete |
| Demo instructions | `docs/demo.md` | Complete |
| Scope distinction | `README.md` | Complete |

## Scope

Out of scope and intentionally not required: full Transformer architecture, multi-head attention, production deployment, frontend/API/database work, large datasets, distributed training, and GPU optimization.

Optional/bonus work not implemented: paper reproduction, LoRA-style adaptation, JAX implementation, and numerical/memory optimization. Manual backpropagation and the d_k scaling study are implemented as isolated bonus experiments.
