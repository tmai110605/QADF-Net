"""
cmig_qadf/models/qc_sca.py
==========================
Quality-Conditioned Spatial Cross-Attention (QC-SCA).
Supports full ablation mechanism variants (Ablation E5):
  - none       : Standard cross-attention (no quality conditioning)
  - scale_k    : Scale Keys by quality q
  - scale_v    : Scale Values by quality q (satisfies Proposition 1)
  - bias_logq  : Logit bias beta * log(q) with SINK TOKEN ("abstention mechanism")
  - all        : Full combination of K + V scaling + Sink Logit Bias
Ensures mathematical correctness and empirical verification of the theoretical upper bound.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class QualityConditionedSpatialCrossAttentionBlock(nn.Module):
    """
    Unidirectional QC-SCA block: Query from View A attends to Key/Value from View B conditioned on q_B.
    """
    def __init__(
        self,
        token_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        qc_mode: str = "all",       # "none", "scale_k", "scale_v", "bias_logq", "all"
        use_sink_token: bool = True,
        q_min: float = 1e-3,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qc_mode = qc_mode
        self.use_sink_token = use_sink_token and (qc_mode in ["bias_logq", "all"])
        self.q_min = q_min

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=False)

        # Parameters for Sink Token mechanism
        if self.use_sink_token:
            # Learned logit bias for sink token: (1, num_heads, 1, 1)
            self.sink_logit = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            # Temperature / scaling factor beta for log(q): initialized to 1.0
            self.beta = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        feat_q: torch.Tensor,       # (B, M, d) from View A
        feat_kv: torch.Tensor,      # (B, M, d) from View B
        q_other: torch.Tensor,      # (B, 1) quality score of View B ∈ [q_min, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            out: (B, M, d) representation after residual + LayerNorm
            attn_weights: (B, num_heads, M, M) attention weight matrix over View B
            diag: dict containing diagnostic measurements (attention norm, theoretical bound)
        """
        B, M, _ = feat_q.shape
        q_clamped = torch.clamp(q_other, min=self.q_min, max=1.0).view(B, 1, 1, 1)

        # Multi-head projection: (B, num_heads, M, head_dim)
        Q = self.q_proj(feat_q).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(feat_kv).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(feat_kv).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Scale-K mechanism
        if self.qc_mode in ["scale_k", "all"]:
            K = K * q_clamped

        # 2. Scale-V mechanism (Proposition 1)
        if self.qc_mode in ["scale_v", "all"]:
            V_used = V * q_clamped
        else:
            V_used = V

        # Compute dot-product logits: (B, num_heads, M, M)
        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 3. Logit bias mechanism with Sink Token
        if self.use_sink_token:
            # Add beta * log(q) to logits of opposite view tokens
            log_q = torch.log(q_clamped)
            logits_biased = logits + torch.abs(self.beta) * log_q

            # Append sink token column: (B, num_heads, M, 1)
            sink_col = self.sink_logit.expand(B, self.num_heads, M, 1)
            logits_all = torch.cat([logits_biased, sink_col], dim=-1) # (B, num_heads, M, M+1)

            attn_probs_all = F.softmax(logits_all, dim=-1)
            attn_weights = attn_probs_all[..., :M] # Attention mass allocated to opposite view
            attn_sink = attn_probs_all[..., M:]    # Attention mass absorbed by sink token
        else:
            if self.qc_mode == "bias_logq":
                # Pure bias without sink (used in unit test to verify Softmax No-Op)
                log_q = torch.log(q_clamped)
                logits = logits + log_q
            attn_weights = F.softmax(logits, dim=-1)
            attn_sink = torch.zeros(B, self.num_heads, M, 1, device=logits.device)

        attn_weights_drop = self.dropout(attn_weights)

        # Aggregate Value vectors: (B, num_heads, M, head_dim)
        attn_out_heads = torch.matmul(attn_weights_drop, V_used)

        # Reshape to (B, M, d)
        attn_out_heads = attn_out_heads.transpose(1, 2).contiguous().view(B, M, self.token_dim)
        raw_attn = self.out_proj(attn_out_heads)

        # Residual + LayerNorm
        out = self.norm(feat_q + raw_attn)

        # Measurements for empirical verification of Proposition 1
        with torch.no_grad():
            raw_attn_norm = torch.norm(raw_attn, p=2, dim=-1).mean().item()
            # Upper bound according to Proposition 1: q * max_j ||W_v f_j||
            v_norms = torch.norm(V.transpose(1, 2).contiguous().view(B, M, self.token_dim), p=2, dim=-1)
            max_v_norm = v_norms.max(dim=-1)[0].mean().item()
            if self.qc_mode in ["scale_v", "all"]:
                bound = float(q_clamped.mean().item() * max_v_norm)
            elif self.use_sink_token:
                mass = attn_weights.sum(dim=-1).mean().item()
                bound = float(mass * max_v_norm)
            else:
                bound = float(max_v_norm)

        diag = {
            "measured_norm": raw_attn_norm,
            "theoretical_bound": max(bound, 1e-6),
            "ratio": raw_attn_norm / max(bound, 1e-6),
            "sink_mass": attn_sink.mean().item() if self.use_sink_token else 0.0,
            "cross_view_mass": attn_weights.sum(dim=-1).mean().item(),
        }

        return out, attn_weights, diag


class QualityConditionedSpatialCrossAttention(nn.Module):
    """
    Bidirectional QC-SCA module (1<-2 and 2<-1) with consensus pooling.
    """
    def __init__(
        self,
        token_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        qc_mode: str = "all",
        use_sink_token: bool = True,
        q_min: float = 1e-3,
    ):
        super().__init__()
        self.qc_mode = qc_mode
        self.block12 = QualityConditionedSpatialCrossAttentionBlock(
            token_dim=token_dim, num_heads=num_heads, dropout=dropout,
            qc_mode=qc_mode, use_sink_token=use_sink_token, q_min=q_min
        )
        self.block21 = QualityConditionedSpatialCrossAttentionBlock(
            token_dim=token_dim, num_heads=num_heads, dropout=dropout,
            qc_mode=qc_mode, use_sink_token=use_sink_token, q_min=q_min
        )

    def forward(
        self,
        f1: torch.Tensor,       # (B, M, d)
        f2: torch.Tensor,       # (B, M, d)
        q1: torch.Tensor,       # (B, 1)
        q2: torch.Tensor,       # (B, 1)
    ) -> Dict[str, torch.Tensor]:
        # 1. Bidirectional attention
        f1_enh, attn12, diag12 = self.block12(f1, f2, q2)
        f2_enh, attn21, diag21 = self.block21(f2, f1, q1)

        # 2. Spatial pooling per view to d-dimensional vectors
        p1 = f1_enh.mean(dim=1)  # (B, d)
        p2 = f2_enh.mean(dim=1)  # (B, d)

        # 3. Consensus pooling with normalized quality weights
        eps = 1e-5
        w1 = q1 / (q1 + q2 + eps)
        w2 = q2 / (q1 + q2 + eps)
        f_cons = w1 * p1 + w2 * p2  # (B, d)

        return {
            "f_cons": f_cons,
            "f1_enh": f1_enh,
            "f2_enh": f2_enh,
            "p1": p1,
            "p2": p2,
            "attn12": attn12,       # Used in E7
            "attn21": attn21,
            "diag12": diag12,       # Used in Proposition 1 verification
            "diag21": diag21,
        }
