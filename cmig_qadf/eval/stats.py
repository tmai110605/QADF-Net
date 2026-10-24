"""
cmig_qadf/eval/stats.py
=======================
Inferential statistics and hypothesis testing suite (Section 6.3):
  1. Patient-level Clustered Bootstrap (B=2000): Identical resampled patient indices across all methods (Paired).
  2. Percentile 95% Confidence Intervals [2.5%, 97.5%].
  3. Two-sided paired p-values from empirical Bootstrap difference distributions.
  4. Holm-Bonferroni step-down Family-Wise Error Rate (FWER) adjustment across primary testing hypotheses.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any


def get_patient_clusters(patient_ids: List[str]) -> Dict[str, List[int]]:
    """Cluster sample indices by unique patient identifier."""
    clusters = {}
    for idx, pid in enumerate(patient_ids):
        pid_str = str(pid)
        if pid_str not in clusters:
            clusters[pid_str] = []
        clusters[pid_str].append(idx)
    return clusters


def generate_bootstrap_replicates(
    unique_patients: List[str],
    clusters: Dict[str, List[int]],
    n_bootstraps: int = 2000,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Pre-generate resampled sample indices for B bootstrap iterations.
    Shared identically across ALL evaluated methods to ensure strict pairing.
    """
    rng = np.random.RandomState(seed)
    n_p = len(unique_patients)
    replicates = []

    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=n_p, replace=True)
        sample_indices = []
        for p in sampled_patients:
            sample_indices.extend(clusters[p])
        replicates.append(np.array(sample_indices, dtype=int))

    return replicates


def patient_cluster_bootstrap_ci(
    data_dict: Dict[str, Any],
    metric_fn: Callable[[Dict[str, Any], np.ndarray], float],
    replicates: List[np.ndarray],
    ci_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute point estimate and 95% percentile confidence interval via clustered bootstrap.
    """
    scores = []
    for idxs in replicates:
        s = metric_fn(data_dict, idxs)
        if not np.isnan(s):
            scores.append(s)

    if len(scores) == 0:
        base = metric_fn(data_dict, np.arange(len(data_dict["patient_id"])))
        return base, base, base

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    point_est = float(np.mean(scores))
    return point_est, lower, upper


def paired_cluster_bootstrap_test(
    prop_dict: Dict[str, Any],
    base_dict: Dict[str, Any],
    metric_fn: Callable[[Dict[str, Any], np.ndarray], float],
    replicates: List[np.ndarray],
    ci_level: float = 0.95
) -> Dict[str, float]:
    """
    Paired bootstrap hypothesis test:
    Computes Delta = Metric(QC-SCA) - Metric(Baseline) across identical bootstrap replicates.
    """
    deltas = []
    for idxs in replicates:
        s_prop = metric_fn(prop_dict, idxs)
        s_base = metric_fn(base_dict, idxs)
        if not np.isnan(s_prop) and not np.isnan(s_base):
            deltas.append(s_prop - s_base)

    if len(deltas) == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    deltas = np.array(deltas)
    mean_diff = float(np.mean(deltas))
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(deltas, alpha * 100))
    ci_upper = float(np.percentile(deltas, (1.0 - alpha) * 100))

    # Two-sided empirical p-value
    p_le = np.mean(deltas <= 0.0)
    p_ge = np.mean(deltas >= 0.0)
    p_val = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def holm_bonferroni_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Holm-Bonferroni step-down correction for multiple testing.
    Input: dict {method_name: p_value}
    Output: dict {method_name: p_holm}
    """
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0

    for rank, (name, p) in enumerate(items):
        multiplier = m - rank
        adj_p = min(1.0, multiplier * p)
        running_max = max(running_max, adj_p)
        adjusted[name] = min(1.0, running_max)

    return adjusted
