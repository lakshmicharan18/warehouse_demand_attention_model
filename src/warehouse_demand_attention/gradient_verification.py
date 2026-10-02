"""Finite-difference verification of autograd gradients for self-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from warehouse_demand_attention.attention import ScaledDotProductSelfAttention


LossFunction = Callable[[ScaledDotProductSelfAttention, torch.Tensor], torch.Tensor]
DEFAULT_EPSILON = 1e-6
DEFAULT_ABSOLUTE_TOLERANCE = 1e-7
DEFAULT_RELATIVE_TOLERANCE = 1e-5
DEFAULT_INDICES = ((0, 0), (0, 1), (1, 0))


@dataclass(frozen=True)
class GradientCheckResult:
    """One comparison between autograd and central-difference gradients."""

    parameter_name: str
    index: tuple[int, int]
    autograd_gradient: float
    numerical_gradient: float
    absolute_difference: float
    relative_difference: float


@dataclass(frozen=True)
class GradientVerificationReport:
    """All selected gradient comparisons from one deterministic verification run."""

    results: tuple[GradientCheckResult, ...]
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float

    @property
    def maximum_absolute_difference(self) -> float:
        return max(result.absolute_difference for result in self.results)

    @property
    def maximum_relative_difference(self) -> float:
        return max(result.relative_difference for result in self.results)

    @property
    def passed(self) -> bool:
        return all(
            result.absolute_difference <= self.absolute_tolerance
            or result.relative_difference <= self.relative_tolerance
            for result in self.results
        )


def squared_mean_output_loss(
    attention: ScaledDotProductSelfAttention, X: torch.Tensor
) -> torch.Tensor:
    """Scalar loss used consistently for autograd and finite differences."""
    output, _ = attention(X)
    return output.square().mean()


def make_verification_example() -> tuple[ScaledDotProductSelfAttention, torch.Tensor]:
    """Create the small, deterministic float64 attention example used for checks."""
    torch.manual_seed(123)
    attention = ScaledDotProductSelfAttention(input_dim=2, d_k=2, d_v=2).to(torch.float64)
    X = torch.tensor(
        [[[0.25, -0.50], [1.00, 0.75], [-0.25, 0.50]]], dtype=torch.float64
    )
    return attention, X


def central_difference_gradient(
    attention: ScaledDotProductSelfAttention,
    X: torch.Tensor,
    parameter: torch.nn.Parameter,
    index: tuple[int, int],
    epsilon: float = DEFAULT_EPSILON,
    loss_function: LossFunction = squared_mean_output_loss,
) -> float:
    """Approximate one parameter derivative with central finite differences.

    The parameter is restored even if a loss evaluation raises an exception.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    with torch.no_grad():
        original = parameter[index].item()
        try:
            parameter[index] = original + epsilon
            loss_plus = loss_function(attention, X).item()
            parameter[index] = original - epsilon
            loss_minus = loss_function(attention, X).item()
        finally:
            parameter[index] = original
    return (loss_plus - loss_minus) / (2 * epsilon)


def verify_gradients(
    attention: ScaledDotProductSelfAttention | None = None,
    X: torch.Tensor | None = None,
    epsilon: float = DEFAULT_EPSILON,
    selected_indices: Iterable[tuple[int, int]] = DEFAULT_INDICES,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> GradientVerificationReport:
    """Compare selected Q/K/V autograd entries with independent finite differences."""
    if attention is None or X is None:
        attention, X = make_verification_example()
    if X.dtype != torch.float64:
        raise ValueError("gradient verification requires float64 input")

    attention.zero_grad(set_to_none=True)
    loss = squared_mean_output_loss(attention, X)
    loss.backward()

    indices = tuple(selected_indices)
    results: list[GradientCheckResult] = []
    for parameter_name in ("W_Q", "W_K", "W_V"):
        parameter = getattr(attention, parameter_name)
        for index in indices:
            autograd_gradient = parameter.grad[index].item()
            numerical_gradient = central_difference_gradient(
                attention, X, parameter, index, epsilon
            )
            absolute_difference = abs(autograd_gradient - numerical_gradient)
            denominator = max(abs(autograd_gradient), abs(numerical_gradient), 1e-12)
            results.append(
                GradientCheckResult(
                    parameter_name=parameter_name,
                    index=index,
                    autograd_gradient=autograd_gradient,
                    numerical_gradient=numerical_gradient,
                    absolute_difference=absolute_difference,
                    relative_difference=absolute_difference / denominator,
                )
            )

    report = GradientVerificationReport(
        tuple(results), epsilon, absolute_tolerance, relative_tolerance
    )
    if not report.passed:
        raise AssertionError(
            "autograd and numerical gradients exceeded the configured tolerances"
        )
    return report
