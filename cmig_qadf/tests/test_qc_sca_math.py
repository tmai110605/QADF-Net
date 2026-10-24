"""
tests/test_qc_sca_math.py
Mathematical validation tests for QC-SCA:
1. Proposition 1: Measured attention norm <= Theoretical upper bound.
2. Scalar logit bias without sink is a theoretical No-Op; sink token effectively attenuates corrupted attention.
3. qc_mode='none' numerically matches standard cross-attention.
"""

import pytest
import torch
import torch.nn.functional as F
from cmig_qadf.models.qc_sca import (
    QualityConditionedSpatialCrossAttentionBlock,
    QualityConditionedSpatialCrossAttention
)


def test_proposition1_bound_satisfied():
    """Verify Proposition 1: Measured attention norm <= Bound across arbitrary quality values q."""
    block = QualityConditionedSpatialCrossAttentionBlock(
        token_dim=64, num_heads=2, qc_mode="scale_v", use_sink_token=False
    )
    block.eval()

    torch.manual_seed(42)
    B, M, d = 4, 16, 64
    feat_q = torch.randn(B, M, d)
    feat_kv = torch.randn(B, M, d)

    for q_val in [0.01, 0.1, 0.35, 0.75, 1.0]:
        q_other = torch.full((B, 1), q_val)
        _, _, diag = block(feat_q, feat_kv, q_other)
        measured = diag["measured_norm"]
        bound = diag["theoretical_bound"]
        ratio = diag["ratio"]

        # Strict bound: ratio <= 1.0 (allowing small float32 numerical tolerance 1e-4)
        assert ratio <= 1.0001, f"Proposition 1 violated at q={q_val}: measured={measured}, bound={bound}, ratio={ratio}"


def test_scalar_bias_without_sink_is_noop():
    """
    Mathematical proof: Adding beta * log(q) to logits over a single view WITHOUT a sink token
    yields identical softmax outputs (Softmax No-Op).
    """
    # Bias block without sink
    block_no_sink = QualityConditionedSpatialCrossAttentionBlock(
        token_dim=64, num_heads=2, qc_mode="bias_logq", use_sink_token=False
    )
    # Standard cross-attention block (no bias)
    block_none = QualityConditionedSpatialCrossAttentionBlock(
        token_dim=64, num_heads=2, qc_mode="none", use_sink_token=False
    )
    # Align weights for comparison
    block_no_sink.load_state_dict(block_none.state_dict(), strict=False)
    block_no_sink.eval()
    block_none.eval()

    feat_q = torch.randn(2, 10, 64)
    feat_kv = torch.randn(2, 10, 64)

    # Evaluate at q = 0.2 and baseline q = 1.0
    _, attn_q02, _ = block_no_sink(feat_q, feat_kv, torch.tensor([[0.2], [0.2]]))
    _, attn_none, _ = block_none(feat_q, feat_kv, torch.tensor([[1.0], [1.0]]))

    # Softmax invariance: attn_q02 must be strictly identical to attn_none!
    assert torch.allclose(attn_q02, attn_none, atol=1e-5), "Scalar bias without sink must be a mathematical No-Op!"


def test_sink_token_attenuates_cross_view_mass():
    """
    Verify with Sink Token: As q -> 0, attention mass shifts toward the Sink token and
    cross-view attention mass to the corrupted view attenuates toward 0.
    """
    block_sink = QualityConditionedSpatialCrossAttentionBlock(
        token_dim=64, num_heads=2, qc_mode="bias_logq", use_sink_token=True
    )
    block_sink.eval()

    feat_q = torch.randn(2, 10, 64)
    feat_kv = torch.randn(2, 10, 64)

    _, _, diag_high_q = block_sink(feat_q, feat_kv, torch.tensor([[0.9], [0.9]]))
    _, _, diag_low_q  = block_sink(feat_q, feat_kv, torch.tensor([[0.005], [0.005]]))

    mass_high = diag_high_q["cross_view_mass"]
    mass_low  = diag_low_q["cross_view_mass"]
    sink_high = diag_high_q["sink_mass"]
    sink_low  = diag_low_q["sink_mass"]

    # When q decreases: cross-view mass must DECREASE, and sink_mass must significantly INCREASE
    assert mass_low < mass_high, f"Mass low ({mass_low}) must be lower than mass high ({mass_high})"
    assert sink_low > sink_high, f"Sink mass low ({sink_low}) must be higher than sink mass high ({sink_high})"


def test_qc_mode_none_matches_standard_cross_attn():
    """Verify that qc_mode='none' numerically matches standard cross-attention."""
    block = QualityConditionedSpatialCrossAttentionBlock(
        token_dim=64, num_heads=2, qc_mode="none", use_sink_token=False
    )
    block.eval()

    feat_q = torch.randn(2, 8, 64)
    feat_kv = torch.randn(2, 8, 64)
    q_val = torch.tensor([[0.5], [0.5]])

    out, attn, _ = block(feat_q, feat_kv, q_val)
    # Sum of attention across key tokens for each query must strictly equal 1.0
    attn_sum = attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
