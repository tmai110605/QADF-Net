"""
cmig_qadf/models/heads.py
=========================
Fixed ordinal classification head (CORAL - Consistent Rank Logits).
100% shared across all compared methods (identical class, parameter count, and loss).
Enforces structural monotonicity: z_0 >= z_1 >= z_2 >= z_3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CoralOrdinalHead(nn.Module):
    """
    CORAL Head for 5 ICDR DR grades (0-4) with 4 cumulative binary cutpoints.
    Parameters: 1 shared weight vector w ∈ R^d + 4 cutpoint biases (1 base + 3 steps).
    Total parameter count: d + 4.
    """
    def __init__(self, in_features: int = 128, num_classes: int = 5):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1  # 4 cutpoints

        # Shared weight vector: w^T x
        self.fc = nn.Linear(in_features, 1, bias=False)

        # Cutpoint biases: b_0 is base, delta_1..3 are positive steps ensuring z_0 >= z_1 >= z_2 >= z_3
        self.base_bias = nn.Parameter(torch.tensor([1.5]))
        self.deltas = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input:  (B, in_features)
        Output: Dict containing logits, cum_probs, pred, expected_grade, probs
        """
        B = x.shape[0]
        base_logit = self.fc(x)  # (B, 1)

        # Construct cutpoints with structural monotonicity
        # z_0 = base_logit + b_0
        # z_1 = z_0 - softplus(delta_1)
        # z_2 = z_1 - softplus(delta_2)
        # z_3 = z_2 - softplus(delta_3)
        step1 = F.softplus(self.deltas[0])
        step2 = F.softplus(self.deltas[1])
        step3 = F.softplus(self.deltas[2])

        z0 = base_logit + self.base_bias
        z1 = z0 - step1
        z2 = z1 - step2
        z3 = z2 - step3

        logits = torch.cat([z0, z1, z2, z3], dim=-1)  # (B, 4)
        cum_probs = torch.sigmoid(logits)             # (B, 4), strictly monotonically decreasing

        # Discrete prediction: y_hat = sum_k 1[sigma(z_k) > 0.5] ∈ {0, 1, 2, 3, 4}
        pred = (cum_probs > 0.5).sum(dim=-1).long()

        # Continuous expected score E[y] = sum_k sigma(z_k) ∈ [0, 4]
        expected_grade = cum_probs.sum(dim=-1)

        # 5-class probability distribution for Macro-AUC (ensuring exact probability simplex):
        # p_0 = 1 - sigma(z0)
        # p_1 = sigma(z0) - sigma(z1)
        # p_2 = sigma(z1) - sigma(z2)
        # p_3 = sigma(z2) - sigma(z3)
        # p_4 = sigma(z3)
        p0 = 1.0 - cum_probs[:, 0:1]
        p1 = cum_probs[:, 0:1] - cum_probs[:, 1:2]
        p2 = cum_probs[:, 1:2] - cum_probs[:, 2:3]
        p3 = cum_probs[:, 2:3] - cum_probs[:, 3:4]
        p4 = cum_probs[:, 3:4]

        probs = torch.cat([p0, p1, p2, p3, p4], dim=-1)
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return {
            "logits": logits,
            "cum_probs": cum_probs,
            "pred": pred,
            "expected_grade": expected_grade,
            "probs": probs,
        }


class CoralLoss(nn.Module):
    """
    Binary Cross-Entropy (BCE) loss across the 4 cumulative cutpoints of CORAL.
    Binary target for ground-truth label y ∈ {0..4}:
        y_k = 1 if y > k, else 0.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 4)
        target: (B,) long labels in {0, 1, 2, 3, 4}
        """
        B = target.shape[0]
        # Construct cumulative binary targets: (B, 4)
        # thresholds = [0, 1, 2, 3]
        cutpoints = torch.arange(self.num_cutpoints, device=target.device).unsqueeze(0) # (1, 4)
        binary_targets = (target.unsqueeze(-1) > cutpoints).float()                      # (B, 4)

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        return loss
