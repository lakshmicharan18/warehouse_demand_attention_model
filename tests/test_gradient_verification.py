import torch

from warehouse_demand_attention.gradient_verification import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    central_difference_gradient,
    make_verification_example,
    squared_mean_output_loss,
    verify_gradients,
)


def test_numerical_gradient_matches_autograd_for_a_deterministic_element():
    attention, X = make_verification_example()
    loss = squared_mean_output_loss(attention, X)
    loss.backward()
    numerical = central_difference_gradient(attention, X, attention.W_Q, (0, 0))
    assert abs(attention.W_Q.grad[0, 0].item() - numerical) <= DEFAULT_ABSOLUTE_TOLERANCE


def test_selected_wq_gradients_agree_with_autograd():
    report = verify_gradients()
    assert all(result.relative_difference <= DEFAULT_RELATIVE_TOLERANCE for result in report.results if result.parameter_name == "W_Q")


def test_selected_wk_gradients_agree_with_autograd():
    report = verify_gradients()
    assert all(result.relative_difference <= DEFAULT_RELATIVE_TOLERANCE for result in report.results if result.parameter_name == "W_K")


def test_selected_wv_gradients_agree_with_autograd():
    report = verify_gradients()
    assert all(result.relative_difference <= DEFAULT_RELATIVE_TOLERANCE for result in report.results if result.parameter_name == "W_V")


def test_all_errors_are_within_configured_tolerance():
    report = verify_gradients()
    assert report.passed
    assert all(
        result.absolute_difference <= DEFAULT_ABSOLUTE_TOLERANCE
        or result.relative_difference <= DEFAULT_RELATIVE_TOLERANCE
        for result in report.results
    )


def test_parameter_is_restored_after_numerical_perturbation():
    attention, X = make_verification_example()
    original = attention.W_V.detach().clone()
    central_difference_gradient(attention, X, attention.W_V, (1, 0))
    assert torch.equal(attention.W_V.detach(), original)
