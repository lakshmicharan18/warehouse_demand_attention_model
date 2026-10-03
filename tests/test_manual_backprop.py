import torch
from warehouse_demand_attention.manual_backprop import compare_manual_to_autograd, softmax_backward, stable_softmax

def test_manual_projection_gradients_match_autograd():
    result = compare_manual_to_autograd()
    for name in ("dW_Q", "dW_K", "dW_V", "dX"):
        assert torch.allclose(result.manual[name], result.autograd[name], atol=1e-10, rtol=1e-10)
        assert torch.isfinite(result.manual[name]).all()

def test_softmax_backward_matches_autograd():
    scores = torch.tensor([[[0.2, -0.1, 0.5]]], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([[[0.3, -0.4, 0.7]]], dtype=torch.float64)
    attention = stable_softmax(scores)
    (attention * upstream).sum().backward()
    assert torch.allclose(softmax_backward(attention.detach(), upstream), scores.grad, atol=1e-12)
