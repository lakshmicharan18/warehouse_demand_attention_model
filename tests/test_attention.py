import math

import torch

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention


def _attention_and_input():
    torch.manual_seed(7)
    return ScaledDotProductSelfAttention(3, 2, 4), torch.randn(2, 5, 3)


def test_projection_and_score_shapes_are_correct():
    attention, X = _attention_and_input()
    _, _, details = attention.forward_with_details(X)
    assert details["Q"].shape == (2, 5, 2)
    assert details["K"].shape == (2, 5, 2)
    assert details["V"].shape == (2, 5, 4)
    assert details["scores"].shape == (2, 5, 5)


def test_output_and_attention_weight_shapes_are_correct_for_batched_input():
    attention, X = _attention_and_input()
    output, weights = attention(X)
    assert output.shape == (2, 5, 4)
    assert weights.shape == (2, 5, 5)


def test_attention_weight_rows_sum_to_one_and_are_non_negative():
    attention, X = _attention_and_input()
    _, weights = attention(X)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.all(weights >= 0)


def test_large_inputs_produce_finite_outputs():
    attention, _ = _attention_and_input()
    X = torch.full((2, 4, 3), 1e4)
    output, weights = attention(X)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


def test_manual_tiny_example_matches_independent_calculation():
    attention = ScaledDotProductSelfAttention(2, 2, 2)
    with torch.no_grad():
        attention.W_Q.copy_(torch.eye(2))
        attention.W_K.copy_(torch.eye(2))
        attention.W_V.copy_(torch.eye(2))
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    output, weights = attention(X)
    diagonal_weight = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    off_diagonal_weight = 1 / (math.exp(1 / math.sqrt(2)) + 1)
    expected_weights = torch.tensor(
        [[[diagonal_weight, off_diagonal_weight], [off_diagonal_weight, diagonal_weight]]]
    )

    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(output, expected_weights, atol=1e-6)


def test_scaled_scores_use_square_root_of_dk():
    attention = ScaledDotProductSelfAttention(2, 4, 2)
    X = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    _, _, details = attention.forward_with_details(X)
    expected = details["Q"] @ details["K"].transpose(-2, -1) / math.sqrt(4)
    assert torch.allclose(details["scaled_scores"], expected)


def test_all_projection_parameters_receive_finite_gradients():
    attention, X = _attention_and_input()
    output, _ = attention(X)
    output.square().mean().backward()
    for parameter in (attention.W_Q, attention.W_K, attention.W_V):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
