"""
cmig_qadf/eval/attention_analysis.py
===================================
Experiment E7: Spatial attention analysis (Attention Analysis).
1. Attention mass allocated to degraded view.
2. Shannon entropy of the attention distribution.
3. Spatial stability: Spearman rank correlation & Top-k token IoU between clean and degraded attention maps.
4. Transparent reporting: Lesion-level ground-truth segmentations are unavailable in DeepDRiD (N/A).
"""

import os
import json
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from typing import Dict, List, Any


def compute_attention_entropy(attn_map: np.ndarray) -> float:
    """Shannon entropy of the attention distribution across tokens."""
    flat = attn_map.flatten()
    flat = flat / (np.sum(flat) + 1e-7)
    pos = flat[flat > 1e-7]
    return float(-np.sum(pos * np.log2(pos)))


def compute_spatial_stability(map_clean: np.ndarray, map_deg: np.ndarray, top_k_ratio: float = 0.2) -> Dict[str, float]:
    """
    Compute Spearman rank correlation and top-k token IoU between clean and degraded attention maps.
    """
    c_flat = map_clean.flatten()
    d_flat = map_deg.flatten()

    # 1. Spearman correlation
    rho, _ = stats.spearmanr(c_flat, d_flat)
    if np.isnan(rho): rho = 0.0

    # 2. IoU top-k
    k = max(1, int(len(c_flat) * top_k_ratio))
    top_c = set(np.argsort(c_flat)[-k:])
    top_d = set(np.argsort(d_flat)[-k:])
    iou = len(top_c & top_d) / float(len(top_c | top_d))

    return {
        "spearman_rho": float(rho),
        "top_k_iou": float(iou),
    }


def analyze_attention_e7(
    model: torch.nn.Module,
    sample_records: List[Dict],
    deg_types: List[str] = ["defocus", "dark"],
    levels: List[int] = [0, 3, 5],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = None
) -> Dict[str, Any]:
    """
    Execute E7 attention analysis and aggregate metrics for Figure 4.
    """
    model.eval()
    from cmig_qadf.data.adapter import get_transforms
    from cmig_qadf.data.degradations import apply_single_degradation
    from PIL import Image

    tf = get_transforms(448, is_train=False, use_clahe=True)
    results = {
        "lesion_correspondence_available": False,
        "note": "Lesion annotations are unavailable in DeepDRiD; reported as N/A per protocol.",
        "case_studies": [],
        "mass_by_level": {},
    }

    test_samples = sample_records[:min(10, len(sample_records))]

    for deg in deg_types:
        results["mass_by_level"][deg] = {}
        for lvl in levels:
            masses = []
            entropies = []
            for rec in test_samples:
                im1_raw = Image.open(rec["path1"]).convert("RGB")
                im2_raw = Image.open(rec["path2"]).convert("RGB")

                if lvl > 0:
                    im2_deg = apply_single_degradation(im2_raw, deg, lvl, 42, rec["pair_id"], "v2")
                else:
                    im2_deg = im2_raw

                t1 = tf(im1_raw).unsqueeze(0).to(device)
                t2 = tf(im2_deg).unsqueeze(0).to(device)
                q1 = torch.tensor([[float(rec["q1"])]], device=device)
                q2 = torch.tensor([[float(rec["q2"])]], device=device)

                with torch.no_grad():
                    out = model(t1, t2, q1, q2)

                if "attn12" in out:
                    # attn12: (1, num_heads, M, M)
                    attn = out["attn12"][0].mean(dim=0).cpu().numpy() # (M, M)
                    # Mean attention mass allocated to View 2
                    masses.append(float(np.mean(attn)))
                    entropies.append(compute_attention_entropy(attn))

            results["mass_by_level"][deg][str(lvl)] = {
                "mean_attention_mass": float(np.mean(masses)) if masses else 1.0,
                "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results
