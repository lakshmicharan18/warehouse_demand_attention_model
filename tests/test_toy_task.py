import torch

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention
from warehouse_demand_attention.toy_task import (
    ToyAttentionRegressor,
    ToyTaskConfig,
    ToyTrainingConfig,
    generate_toy_data,
    train_toy_model,
)


def test_toy_data_has_expected_shape_and_target_formula_without_noise():
    X, y = generate_toy_data(10, seed=4, noise_std=0.0)
    assert X.shape == (10, 6)
    assert y.shape == (10,)
    assert torch.allclose(y, 0.7 * X[:, 1] + 0.3 * X[:, 4])


def test_toy_data_is_reproducible_with_fixed_seed():
    first_X, first_y = generate_toy_data(10, seed=4)
    second_X, second_y = generate_toy_data(10, seed=4)
    assert torch.equal(first_X, second_X)
    assert torch.equal(first_y, second_y)


def test_model_output_shape_and_custom_attention_usage():
    model = ToyAttentionRegressor()
    predictions, weights = model(torch.randn(4, 6))
    assert predictions.shape == (4,)
    assert weights.shape == (4, 6, 6)
    assert isinstance(model.attention, ScaledDotProductSelfAttention)


def test_forward_and_backward_values_are_finite():
    model = ToyAttentionRegressor()
    predictions, _ = model(torch.randn(4, 6))
    predictions.square().mean().backward()
    assert torch.isfinite(predictions).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_short_training_run_decreases_loss():
    result = train_toy_model(
        ToyTaskConfig(train_samples=256, validation_samples=64, test_samples=64, target_noise_std=0.0),
        ToyTrainingConfig(epochs=30, batch_size=64),
    )
    assert result.training_losses[-1] < result.training_losses[0]
    assert result.gradients_finite
