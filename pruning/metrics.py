"""
Pruning metrics for NetGPT post-training pruning.

Implements three metrics from the literature:
  1. Magnitude  - |W|                          (Han et al., 2016)
  2. Wanda      - |W_ij| * sqrt(sum(X_j^2))    (Sun et al., 2024)
  3. Pruner-Zero - sqrt(|W|^2 * minmax(|G|))   (Dong et al., 2024)

Each metric returns a score tensor of the same shape as the weight matrix.
Higher score = more important weight = should be kept.

Reference implementations:
  - Wanda: https://github.com/locuslab/wanda/blob/main/lib/prune.py
  - Pruner-Zero: https://github.com/pprp/Pruner-Zero (built on Wanda)
"""

import torch


def magnitude_score(weight, **kwargs):
    """
    Magnitude pruning: S(W) = |W|
    
    Source: Han et al. 2016 "Deep Compression", Table 1 of Pruner-Zero.
    Simplest metric — only considers weight size. No calibration needed.
    """
    return weight.abs()


def wanda_score(weight, activation_norms=None, **kwargs):
    """
    Wanda pruning: S(W_ij) = |W_ij| * sqrt(scaler_row_j)
    
    Source: Sun et al. 2024, Table 1 of Pruner-Zero.
    
    In the Wanda codebase (lib/prune.py line ~157):
        W_metric = torch.abs(W.data) * torch.sqrt(scaler_row.reshape((1,-1)))
    where scaler_row accumulates sum(X_j^2) over calibration samples.
    
    Our activation_norms is already sqrt(sum(X_j^2)), matching their formula.
    
    Args:
        weight: (out_features, in_features) weight matrix
        activation_norms: (in_features,) = sqrt(sum of squared activations per dim)
    """
    if activation_norms is None:
        raise ValueError("Wanda requires activation_norms (per-input-dim L2 norms)")
    
    # weight: (out, in), activation_norms: (in,) -> broadcast via unsqueeze
    return weight.abs() * activation_norms.unsqueeze(0)


def pruner_zero_score(weight, gradients=None, **kwargs):
    """
    Pruner-Zero: S(W) = abs(W * W) * mms(G) = W^2 * mms(G)
    
    Source: Dong et al. 2024, Equation 2 (page 6) + actual code tree from
    github.com/pprp/Pruner-Zero README:
    
        { "data": "mul",
          "left":  { "data": "abs", "left": { "data": "mul",
                     "left": {"data":"W"}, "right": {"data":"W"} } },
          "right": { "data": "mms", "left": {"data":"G"} } }
    
    This reads as: abs(W*W) * mms(G) = W^2 * min_max_scale(G)
    
    IMPORTANT distinctions from a naive reading of Eq. 2:
      - NO sqrt: the expression tree has no sqrt at the root
      - mms(G) NOT mms(|G|): applied to raw gradients, not absolute values.
        Negative gradients map near 0, positive near 1. The sign carries
        information about whether increasing the weight helps or hurts.
      - W^2 = abs(W*W): squaring makes abs redundant but it's in the tree
    
    Theoretical justification (Appendix C.4, Eq. 7):
        I(W) = (W x G)^2  =>  importance = weight * gradient interaction
    
    Args:
        weight: (out_features, in_features) weight matrix
        gradients: (out_features, in_features) accumulated gradients from calibration
    """
    if gradients is None:
        raise ValueError("Pruner-Zero requires gradients")
    
    # abs(W * W) = W^2 (squaring penalizes small weights quadratically)
    w_sq = weight * weight  # NOT weight.abs() ** 2 — matches mul(W,W) in tree
    w_sq = w_sq.abs()       # abs() from the tree (redundant since W^2 >= 0)
    
    # mms(G) = min-max scaling of RAW gradients (not |G|)
    # This is operation U10 from Table 10: sc = min-max scale(sa)
    g = gradients  # raw gradients, NOT .abs()
    g_min = g.min()
    g_max = g.max()
    eps = 1e-8  # numerical stability (our addition, prevents div-by-zero)
    g_scaled = (g - g_min) / (g_max - g_min + eps)
    
    # Final score: W^2 * mms(G) — no sqrt
    score = w_sq * g_scaled
    
    return score



def random_score(weight, **kwargs):
    """Baseline: scores aleatoires (ShrinkBench/Blalock). Seed fixe pour reproductibilite."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(42)
    return torch.rand(weight.shape, generator=g).to(weight.device)


# Registry for CLI access
METRICS = {
    "magnitude": magnitude_score,
    "wanda": wanda_score,
    "pruner_zero": pruner_zero_score,
    "random": random_score,
}
