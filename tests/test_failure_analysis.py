import numpy as np
import torch

from warehouse_demand_attention.failure_analysis import absolute_errors, error_percentiles, largest_error_index
from warehouse_demand_attention.warehouse_model import WarehouseDemandAttentionModel


def test_error_calculation_selection_and_percentiles():
    errors = absolute_errors(np.array([2., 9., 4., 1.]), np.array([1., 1., 3., 1.]))
    assert np.array_equal(errors, np.array([1., 8., 1., 0.]))
    assert largest_error_index(errors) == 1
    stats = error_percentiles(errors)
    assert stats["maximum"] == 8.0 and stats["median"] == 1.0


def test_attention_weights_are_finite_normalized_and_model_is_unchanged_by_forward():
    model = WarehouseDemandAttentionModel()
    before = [p.detach().clone() for p in model.parameters()]
    _, weights = model(torch.randn(2, 24))
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(-1), torch.ones(2, 24), atol=1e-6)
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), before))
