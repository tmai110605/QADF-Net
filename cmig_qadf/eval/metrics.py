"""
cmig_qadf/eval/metrics.py
=========================
Standard evaluation metrics: QWK, Accuracy, Macro-AUC, Per-Class Cohen's Kappa.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa for 5-grade DR (0-4)."""
    try:
        val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return 0.0 if np.isnan(val) else float(val)
    except Exception:
        return 0.0


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_auc(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 5) -> float:
    """Macro-AUC (One-vs-Rest)."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        if y_probs.shape[1] == 1:
            return float(roc_auc_score(y_true, y_probs))
        return float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except Exception:
        return 0.5


def compute_per_class_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> dict:
    """Binary Cohen's Kappa per ICDR grade (One-vs-Rest)."""
    kappas = {}
    for c in range(num_classes):
        yt_bin = (y_true == c).astype(int)
        yp_bin = (y_pred == c).astype(int)
        try:
            val = cohen_kappa_score(yt_bin, yp_bin)
            kappas[f"kappa_class_{c}"] = 0.0 if np.isnan(val) else float(val)
        except Exception:
            kappas[f"kappa_class_{c}"] = 0.0
    return kappas
