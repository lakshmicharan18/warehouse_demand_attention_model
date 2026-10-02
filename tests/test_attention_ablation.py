import math

import torch

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention
from warehouse_demand_attention.toy_task import ToyTaskConfig, ToyTrainingConfig, train_toy_model


def _attention_pair():
    torch.manual_seed(10)
    scaled = ScaledDotProductSelfAttention(3, 4, 2)
    unscaled = ScaledDotProductSelfAttention(3, 4, 2, scale_scores=False)
    with torch.no_grad():
        unscaled.W_Q.copy_(scaled.W_Q)
        unscaled.W_K.copy_(scaled.W_K)
        unscaled.W_V.copy_(scaled.W_V)
    return scaled, unscaled, torch.randn(2, 5, 3)


def test_scaled_scores_divide_raw_scores_by_square_root_of_dk():
    scaled, _, X = _attention_pair()
    _, _, details = scaled.forward_with_details(X)
    assert torch.allclose(details["scaled_scores"], details["scores"] / math.sqrt(4))


def test_unscaled_scores_leave_raw_scores_unchanged():
    _, unscaled, X = _attention_pair()
    _, _, details = unscaled.forward_with_details(X)
    assert torch.allclose(details["scaled_scores"], details["scores"])


def test_both_modes_have_expected_shapes_and_valid_probabilities():
    scaled, unscaled, X = _attention_pair()
    for attention in (scaled, unscaled):
        output, weights = attention(X)
        assert output.shape == (2, 5, 2)
        assert weights.shape == (2, 5, 5)
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_scaled_mode_remains_the_default():
    assert ScaledDotProductSelfAttention(2, 4, 2).scale_scores is True


def test_short_training_run_works_for_both_attention_modes():
    task = ToyTaskConfig(train_samples=256, validation_samples=64, test_samples=64, target_noise_std=0.0)
    shared = dict(embedding_dim=16, d_k=16, d_v=16, batch_size=128, epochs=15, model_seed=9)
    scaled = train_toy_model(task, ToyTrainingConfig(**shared, scale_scores=True))
    unscaled = train_toy_model(task, ToyTrainingConfig(**shared, scale_scores=False))
    assert scaled.training_losses[-1] < scaled.training_losses[0]
    assert unscaled.training_losses[-1] < unscaled.training_losses[0]
    assert scaled.gradients_finite and unscaled.gradients_finite
