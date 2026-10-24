"""
cmig_qadf/eval/new_metrics.py
=============================
Novel evaluation metrics proposed in the CMIG paper (Section 6.2):
  1. Fusion Gain = QWK_fusion - QWK_best_single_view
  2. Harm Rate = P(|y_hat_fusion - y| > |y_hat_bestview - y|)
  3. AUDC = Area Under Degradation Curve (normalized by step length)
  4. Feature Drift Index (FDI) = ||f_deg - f_clean|| / ||f_clean||
  5. Prediction Flip Rate (PFR) = Rate of discrete label flips under degradation
"""

import numpy as np
from typing import Dict, Union, List


def compute_fusion_gain(qwk_fusion: float, qwk_best_single: float) -> float:
    """QWK gain achieved by fusion model over the best single view."""
    return float(qwk_fusion - qwk_best_single)


def compute_harm_rate(
    y_true: np.ndarray,
    y_pred_fusion: np.ndarray,
    y_pred_bestview: np.ndarray
) -> Dict[str, float]:
    """
    Proportions of instances where fusion produces harm (Harm Rate), equal error (Tie Rate), or improvement (Help Rate).
    The sum of these three rates strictly equals 1.0.
    """
    yt = np.asarray(y_true)
    yf = np.asarray(y_pred_fusion)
    yb = np.asarray(y_pred_bestview)

    err_fusion   = np.abs(yf - yt)
    err_bestview = np.abs(yb - yt)

    harm_mask = err_fusion > err_bestview
    tie_mask  = err_fusion == err_bestview
    help_mask = err_fusion < err_bestview

    n = max(1, len(yt))
    return {
        "harm_rate": float(np.sum(harm_mask) / n),
        "tie_rate":  float(np.sum(tie_mask) / n),
        "help_rate": float(np.sum(help_mask) / n),
    }


def compute_audc(qwk_curve: Union[List[float], np.ndarray]) -> float:
    """
    Area Under Degradation Curve (AUDC) computed via the trapezoidal rule across degradation steps.
    Normalized by the number of steps so that the metric remains on the [-1, 1] scale.
    """
    curve = np.asarray(qwk_curve, dtype=float)
    if len(curve) < 2:
        return float(curve[0]) if len(curve) == 1 else 0.0

    steps = len(curve) - 1
    # Trapezoidal rule
    trapezoid_sum = 0.0
    for i in range(steps):
        trapezoid_sum += (curve[i] + curve[i + 1]) / 2.0

    audc = trapezoid_sum / float(steps)
    return float(audc)


def compute_fdi(f_cons_clean: np.ndarray, f_cons_degraded: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Feature Drift Index: ||f_deg - f_clean||_2 / ||f_clean||_2.
    """
    c = np.asarray(f_cons_clean, dtype=float)
    d = np.asarray(f_cons_degraded, dtype=float)

    diff_norm = np.linalg.norm(d - c, axis=-1)
    clean_norm = np.linalg.norm(c, axis=-1)
    clean_norm = np.maximum(clean_norm, 1e-6)

    sample_fdi = diff_norm / clean_norm
    return {
        "mean_fdi": float(np.mean(sample_fdi)),
        "std_fdi": float(np.std(sample_fdi)),
        "sample_fdi": sample_fdi,
    }


def compute_pfr(y_pred_clean: np.ndarray, y_pred_degraded: np.ndarray) -> float:
    """Prediction Flip Rate: Ratio of samples changing discrete prediction grade when degraded."""
    yc = np.asarray(y_pred_clean)
    yd = np.asarray(y_pred_degraded)
    n = max(1, len(yc))
    flips = np.sum(yc != yd)
    return float(flips / n)
