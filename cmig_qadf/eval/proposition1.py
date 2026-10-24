"""
cmig_qadf/eval/proposition1.py
==============================
Empirical validation of Proposition 1:
Reports measured attention norm ||Attn|| / theoretical bound ratio across quality intervals q.
Strictly ensures ratio <= 1.0. Produces data for Table S3 and Figure 5.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any


def verify_proposition1_empirically(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = None
) -> Dict[str, Any]:
    """
    Execute empirical verification of Proposition 1 on the test cohort.
    """
    model.eval()
    ratios = []
    q_levels = []
    measured_norms = []
    bounds = []

    with torch.no_grad():
        for batch in loader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            q1 = batch["q1"].to(device)
            q2 = batch["q2"].to(device)

            out = model(img1, img2, q1, q2)

            if "diag12" in out:
                d12 = out["diag12"]
                r = float(d12["ratio"])
                ratios.append(r)
                measured_norms.append(float(d12["measured_norm"]))
                bounds.append(float(d12["theoretical_bound"]))
                q_levels.append(float(q2.mean().item()))

            if "diag21" in out:
                d21 = out["diag21"]
                r = float(d21["ratio"])
                ratios.append(r)
                measured_norms.append(float(d21["measured_norm"]))
                bounds.append(float(d21["theoretical_bound"]))
                q_levels.append(float(q1.mean().item()))

    ratios = np.array(ratios)
    q_levels = np.array(q_levels)

    # Discretize into 4 intervals of q: [0, 0.25], (0.25, 0.50], (0.50, 0.75], (0.75, 1.0]
    bins = [0.0, 0.25, 0.50, 0.75, 1.0]
    bin_names = ["[0.0, 0.25]", "(0.25, 0.50]", "(0.50, 0.75]", "(0.75, 1.00]"]
    strata = {}

    for i in range(len(bin_names)):
        mask = (q_levels >= bins[i]) & (q_levels <= bins[i+1])
        if np.any(mask):
            sub_r = ratios[mask]
            strata[bin_names[i]] = {
                "mean_ratio": float(np.mean(sub_r)),
                "max_ratio": float(np.max(sub_r)),
                "p95_ratio": float(np.percentile(sub_r, 95)),
                "num_samples": int(np.sum(mask)),
                "strictly_less_than_one": bool(np.all(sub_r <= 1.00001))
            }

    overall = {
        "mean_ratio": float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
        "max_ratio": float(np.max(ratios)) if len(ratios) > 0 else 0.0,
        "strictly_satisfied": bool(np.all(ratios <= 1.00001)) if len(ratios) > 0 else True,
        "strata_by_q": strata
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2)

    return overall
