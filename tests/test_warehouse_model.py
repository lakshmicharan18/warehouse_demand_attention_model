import numpy as np
import torch

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention
from warehouse_demand_attention.baselines import last_observation_baseline, moving_average_baseline
from warehouse_demand_attention.training import DemandNormalizer, WarehouseTrainingConfig, prepare_warehouse_data, train_warehouse_model
from warehouse_demand_attention.warehouse_model import WarehouseDemandAttentionModel


def test_model_shape_positions_attention_and_finite_forward_backward():
    model = WarehouseDemandAttentionModel()
    output, weights = model(torch.randn(3, 24))
    output.square().mean().backward()
    assert output.shape == (3,) and weights.shape == (3, 24, 24)
    assert model.position_embeddings.shape[0] == 24
    assert isinstance(model.attention, ScaledDotProductSelfAttention)
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_baselines_return_expected_values():
    X = np.array([[1., 2., 3.], [4., 5., 6.]])
    assert np.array_equal(last_observation_baseline(X), np.array([3., 6.]))
    assert np.array_equal(moving_average_baseline(X), np.array([2., 5.]))


def test_normalizer_uses_training_values_only_and_split_is_ordered():
    data = prepare_warehouse_data()
    normalizer = DemandNormalizer.fit(np.array([[1., 3.], [5., 7.]]))
    assert normalizer.mean == 4.0
    assert data.train_X.shape[0] == 3007 and data.validation_X.shape[0] == 644 and data.test_X.shape[0] == 645
    assert data.normalizer.mean != data.test_X.mean()


def test_short_training_decreases_loss():
    result = train_warehouse_model(WarehouseTrainingConfig(embedding_dim=8, d_k=8, d_v=8, epochs=15, batch_size=128))
    assert result.training_losses[-1] < result.training_losses[0]
