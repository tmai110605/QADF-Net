"""
tests/test_coral_head_and_loss.py
Structural monotonicity and architectural fairness tests for CORAL Head.
"""

import pytest
import torch
from cmig_qadf.models.heads import CoralOrdinalHead, CoralLoss
from cmig_qadf.models.dualview_model import DualViewGradingModel


def test_coral_monotonicity_and_simplex():
    head = CoralOrdinalHead(in_features=128, num_classes=5)
    x = torch.randn(20, 128)

    out = head(x)
    logits = out["logits"]        # (B, 4)
    cum_probs = out["cum_probs"]  # (B, 4)
    pred = out["pred"]            # (B,)
    exp_grade = out["expected_grade"] # (B,)
    probs = out["probs"]          # (B, 5)

    # 1. Logits and cumulative probabilities must be strictly monotonically decreasing
    for b in range(20):
        for k in range(3):
            assert logits[b, k] >= logits[b, k + 1], f"Logit violated monotonicity at sample {b}, cutpoint {k}"
            assert cum_probs[b, k] >= cum_probs[b, k + 1], f"Cumulative prob violated monotonicity at sample {b}, cutpoint {k}"

    # 2. Discrete predictions must fall in {0, 1, 2, 3, 4}
    assert (pred >= 0).all() and (pred <= 4).all()
    assert (exp_grade >= 0.0).all() and (exp_grade <= 4.0).all()

    # 3. 5-class probability distribution must reside on the exact simplex (sum = 1.0, non-negative)
    assert (probs >= 0.0).all()
    probs_sum = probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_coral_loss_gradient():
    loss_fn = CoralLoss(num_classes=5)
    logits = torch.randn(10, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0, 4, 3], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None and not torch.isnan(logits.grad).any()


def test_identical_head_architecture_across_all_methods():
    """Verify that all comparative models strictly share identical CORAL head class and parameter count."""
    methods = ["qc_sca", "concat", "average", "max", "q_weighted", "cross_attn", "single_v1"]
    ref_head_params = None

    for m in methods:
        model = DualViewGradingModel(backbone_name="efficientnet_b4", fusion_type=m, pretrained=False)
        assert isinstance(model.head, CoralOrdinalHead), f"Model {m} does not utilize CoralOrdinalHead!"
        head_p_count = sum(p.numel() for p in model.head.parameters())
        if ref_head_params is None:
            ref_head_params = head_p_count
        else:
            assert head_p_count == ref_head_params, f"Head parameter count of {m} ({head_p_count}) differs from reference ({ref_head_params})!"
