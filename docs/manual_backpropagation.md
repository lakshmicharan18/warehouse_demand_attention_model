# Manual Backpropagation Through Attention

This optional bonus uses a fixed float64 input `X` of shape `[1, 3, 2]` with `d_k=d_v=2` and loss `L = mean(Y²)`.

## Forward equations

`Q=XW_Q`, `K=XW_K`, `V=XW_V`; each is `[1,3,2]`. `S=QKᵀ/sqrt(d_k)` and attention `A=softmax(S)` are `[1,3,3]`; `Y=AV` is `[1,3,2]`.

## Step 1 — backward through `Y=AV`

Given `dY = 2Y / numel(Y)`, matrix multiplication gives `dA=dYVᵀ` and `dV=AᵀdY`. These say how changing weights or values changes the output.

## Step 2 — backward through softmax

For each attention row `a`, the Jacobian is `J=diag(a)-aaᵀ`. Rather than materializing `J`, the equivalent vector form is `dS=a ⊙ (dA - sum(dA⊙a))`, with the sum taken across the key dimension.

## Step 3 — backward through scaled `QKᵀ`

`dQ=dS K/sqrt(d_k)` and `dK=dSᵀ Q/sqrt(d_k)`. The scale appears in the backward equations because it appeared in the forward score calculation.

## Step 4 — backward through projections

`dW_Q=XᵀdQ`, `dW_K=XᵀdK`, and `dW_V=XᵀdV`, summing over batch items. `dX` is the sum of contributions from Q, K, and V projections.

## Step 5 — comparison with autograd

`experiments/manual_backprop_experiment.py` computes the manual tensors, then separately uses PyTorch autograd as a reference. This complements finite differences: manual backpropagation checks the explicit chain rule, autograd checks framework differentiation, and central differences provide an independent numerical approximation.
