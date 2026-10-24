"""
cmig_qadf/models/fusion_baselines.py
====================================
Comparative fusion baselines (Fusion Baselines 1-5).
All baselines strictly share:
  - Identical backbone and token representation (B × M × d)
  - Identical consensus representation vector f_cons ∈ R^d
  - Identical fixed CORAL Head layer and BCE loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ConcatFusion(nn.Module):
    """Concatenate spatial pooled representations of both views and project linearly to d."""
    def __init__(self, token_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(token_dim * 2, token_dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)  # (B, d)
        p2 = f2.mean(dim=1)  # (B, d)
        cat = torch.cat([p1, p2], dim=-1)
        f_cons = self.proj(cat)
        return {"f_cons": f_cons, "p1": p1, "p2": p2}


class AverageFusion(nn.Module):
    """Unweighted arithmetic mean (50/50)."""
    def __init__(self, token_dim: int = 128):
        super().__init__()

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)
        p2 = f2.mean(dim=1)
        f_cons = 0.5 * (p1 + p2)
        return {"f_cons": f_cons, "p1": p1, "p2": p2}


class MaxFusion(nn.Module):
    """Element-wise maximum between the two views."""
    def __init__(self, token_dim: int = 128):
        super().__init__()

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)
        p2 = f2.mean(dim=1)
        f_cons = torch.maximum(p1, p2)
        return {"f_cons": f_cons, "p1": p1, "p2": p2}


class QWeightedAverageFusion(nn.Module):
    """
    Quality-weighted average: w_i = q_i / (q_1 + q_2).
    Critical comparative baseline against contribution C1 in the CMIG paper!
    """
    def __init__(self, token_dim: int = 128):
        super().__init__()

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)
        p2 = f2.mean(dim=1)
        eps = 1e-5
        w1 = q1 / (q1 + q2 + eps)  # (B, 1)
        w2 = q2 / (q1 + q2 + eps)
        f_cons = w1 * p1 + w2 * p2
        return {"f_cons": f_cons, "p1": p1, "p2": p2, "w1": w1, "w2": w2}


class CoAttentionBilinearFusion(nn.Module):
    """Co-attention / Bilinear interaction between spatial pooled representations."""
    def __init__(self, token_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.token_dim = token_dim
        self.bilinear = nn.Bilinear(token_dim, token_dim, token_dim)
        self.proj = nn.Linear(token_dim * 2, token_dim)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)  # (B, d)
        p2 = f2.mean(dim=1)  # (B, d)
        bilin = F.relu(self.bilinear(p1, p2))
        f_cons = self.proj(torch.cat([0.5 * (p1 + p2), bilin], dim=-1))
        return {"f_cons": f_cons, "p1": p1, "p2": p2}


class SingleViewSelectorFusion(nn.Module):
    """
    Selects View 1 only, View 2 only, or best view dynamically based on q/q_hat.
    """
    def __init__(self, mode: str = "v1"):
        super().__init__()
        self.mode = mode  # "v1", "v2", "best_q"

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)
        p2 = f2.mean(dim=1)

        if self.mode == "v1":
            f_cons = p1
        elif self.mode == "v2":
            f_cons = p2
        else: # "best_q"
            # Select view with higher quality per sample
            mask = (q1 >= q2).float()  # (B, 1)
            f_cons = mask * p1 + (1.0 - mask) * p2

        return {"f_cons": f_cons, "p1": p1, "p2": p2}
