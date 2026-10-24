"""
tests/test_split_no_leakage.py
Zero patient leakage verification test across splits (0 overlap).
"""

import os
import pytest
from cmig_qadf.data.split import get_patient_splits


def test_official_split_zero_leakage():
    deepdrid_dir = os.path.join("DeepDRiD", "regular_fundus_images")
    if not os.path.exists(deepdrid_dir):
        pytest.skip("DeepDRiD directory not found on test machine")

    splits = get_patient_splits(deepdrid_dir, protocol="official")

    s_train = set(splits["train"])
    s_val   = set(splits["val"])
    s_test  = set(splits["test"])

    # Assert 0 intersection
    assert len(s_train & s_val) == 0, f"Patient leakage between train and val: {s_train & s_val}"
    assert len(s_train & s_test) == 0, f"Patient leakage between train and test: {s_train & s_test}"
    assert len(s_val & s_test) == 0, f"Patient leakage between val and test: {s_val & s_test}"
    assert len(s_train) >= 30, "Insufficient train patient count"
    assert "hash" in splits, "Missing SHA-256 hash in split manifest"
