import math
import torch
from warehouse_demand_attention.dimension_scaling_experiment import DIMENSIONS, attention_statistics, gradient_norm, logits, run_trial

def test_dimensions_and_logit_formulas():
    assert DIMENSIONS == (4,8,16,32,64,128)
    Q=torch.randn(2,3,4,dtype=torch.float64); K=torch.randn(2,3,4,dtype=torch.float64)
    raw, scaled=logits(Q,K)
    assert torch.allclose(raw,Q@K.transpose(-2,-1))
    assert torch.allclose(scaled,raw/math.sqrt(4))

def test_softmax_entropy_gradients_and_theory_ratio_are_finite():
    Q=torch.randn(64,24,32,dtype=torch.float64); K=torch.randn(64,24,32,dtype=torch.float64); V=torch.randn(64,24,32,dtype=torch.float64)
    raw,_=logits(Q,K); maximum, entropy=attention_statistics(raw)
    assert 0 <= maximum <= 1 and math.isfinite(entropy)
    assert math.isfinite(gradient_norm(Q,K,V,True))
    ratios=[run_trial(d, 9, batch_size=256).raw_std/math.sqrt(d) for d in (4,32,128)]
    assert max(ratios)-min(ratios) < 0.08
