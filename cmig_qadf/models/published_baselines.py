"""
cmig_qadf/models/published_baselines.py
=======================================
Registry for published multi-view/dual-view DR methods from the literature.
RESEARCH NOTES:
  - Strictly avoid fabricating method names or citations that have not been verified.
  - The slots below are placeholders reserved for authors to populate upon finalized external baselines.
  - Includes wrappers for internal dual-view baselines from the sister codebase (Dual-view Avg, Dual-view DS).
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class InternalDualViewAverageBaseline(nn.Module):
    """
    Internal baseline: Dual-view Average Fusion from the sister codebase (B3).
    """
    def __init__(self, token_dim: int = 128):
        super().__init__()
        self.source = "Internal Sister Baseline B3 (Dual-view Average)"

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        p1 = f1.mean(dim=1)
        p2 = f2.mean(dim=1)
        return {"f_cons": 0.5 * (p1 + p2), "p1": p1, "p2": p2}


class PublishedDualViewBaselineSlot(nn.Module):
    """
    TODO: Placeholder for external literature methods to be added by authors.
    """
    def __init__(self, method_tag: str = "TBD_PUBLISHED_METHOD"):
        super().__init__()
        self.method_tag = method_tag

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"Method '{self.method_tag}' is currently a placeholder. "
            "Implementation and publication citations will be supplied when available."
        )


PUBLISHED_BASELINES_REGISTRY = {
    "internal_dual_avg": InternalDualViewAverageBaseline,
    # TODO: Add external literature methods here:
    # "method_author_year": PublishedDualViewBaselineSlot,
}
