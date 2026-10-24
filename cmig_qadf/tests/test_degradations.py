"""
tests/test_degradations.py
Determinism and physical monotonicity tests for the 7-type degradation suite.
"""

import pytest
import numpy as np
from PIL import Image
from cmig_qadf.data.degradations import (
    DEGRADATION_TYPES,
    apply_single_degradation,
    degrade_pair
)
from cmig_qadf.data.degradation_calibration import run_physical_monotonicity_test


def test_degradations_determinism():
    """Verify that identical seed, sample_id, and level produce 100% deterministic outputs."""
    # Synthetic random image
    arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    for dt in DEGRADATION_TYPES:
        out1 = apply_single_degradation(img, dt, level=3, seed=42, sample_id="sample_001", view="v1")
        out2 = apply_single_degradation(img, dt, level=3, seed=42, sample_id="sample_001", view="v1")
        out3 = apply_single_degradation(img, dt, level=3, seed=99, sample_id="sample_001", view="v1")

        arr1 = np.array(out1)
        arr2 = np.array(out2)
        arr3 = np.array(out3)

        # Identical seed must yield identical arrays
        assert np.array_equal(arr1, arr2), f"Determinism failure for {dt}"


def test_degradations_monotonicity():
    """Verify monotonic metric variation across severity levels 0 to 5."""
    # Synthetic patterned image
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[30:90, 30:90] = 200
    arr[50:70, 50:70] = 50
    img = Image.fromarray(arr)

    mono_results = run_physical_monotonicity_test(img, seed=42)
    for dt, res in mono_results.items():
        assert res["is_monotonic"], f"Degradation {dt} failed monotonicity requirement: {res['values']}"
