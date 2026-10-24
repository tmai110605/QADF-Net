"""
tests/test_stats_bootstrap.py
Reproducibility tests for patient-level bootstrap and mathematical validity of Holm-Bonferroni correction.
"""

import pytest
import numpy as np
from cmig_qadf.eval.stats import (
    get_patient_clusters,
    generate_bootstrap_replicates,
    holm_bonferroni_correction
)


def test_patient_bootstrap_reproducibility():
    """Verify that identical random seeds produce 100% reproducible resample indices."""
    patients = ["p1", "p2", "p3", "p4", "p5"]
    patient_ids = ["p1", "p1", "p2", "p2", "p3", "p4", "p5", "p5"]
    clusters = get_patient_clusters(patient_ids)

    rep1 = generate_bootstrap_replicates(patients, clusters, n_bootstraps=50, seed=42)
    rep2 = generate_bootstrap_replicates(patients, clusters, n_bootstraps=50, seed=42)

    for i in range(len(rep1)):
        assert np.array_equal(rep1[i], rep2[i]), f"Replicate mismatch at iteration {i}"


def test_holm_bonferroni_reference_match():
    """
    Verify Holm-Bonferroni step-down adjusted p-values against standard analytical formulation:
    Raw p-values:
      m1: 0.005 (rank 1, M=4 -> 0.005 * 4 = 0.02)
      m2: 0.015 (rank 2, M=3 -> 0.015 * 3 = 0.045)
      m3: 0.030 (rank 3, M=2 -> 0.030 * 2 = 0.06)
      m4: 0.100 (rank 4, M=1 -> 0.100 * 1 = 0.10)
    """
    raw_p = {
        "m1": 0.005,
        "m2": 0.015,
        "m3": 0.030,
        "m4": 0.100,
    }

    adj_p = holm_bonferroni_correction(raw_p)

    assert pytest.approx(adj_p["m1"], rel=1e-5) == 0.02
    assert pytest.approx(adj_p["m2"], rel=1e-5) == 0.045
    assert pytest.approx(adj_p["m3"], rel=1e-5) == 0.06
    assert pytest.approx(adj_p["m4"], rel=1e-5) == 0.10
