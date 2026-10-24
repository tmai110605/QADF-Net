"""
tests/test_new_metrics.py
Unit tests verifying proposed metrics against exact hand-calculated analytical examples.
"""

import pytest
import numpy as np
from cmig_qadf.eval.new_metrics import (
    compute_fusion_gain,
    compute_harm_rate,
    compute_audc,
    compute_fdi,
    compute_pfr
)


def test_fusion_gain_hand_calc():
    qwk_f = 0.88
    qwk_b = 0.82
    expected = 0.06
    actual = compute_fusion_gain(qwk_f, qwk_b)
    assert pytest.approx(actual, rel=1e-5) == expected


def test_harm_rate_hand_calc():
    """
    4-sample hand-calculated example:
    y_true          = [2, 1, 3, 0]
    y_pred_fusion   = [2, 3, 2, 0]  (Errors: [0, 2, 1, 0])
    y_pred_bestview = [2, 1, 1, 1]  (Errors: [0, 0, 2, 1])
    Error comparisons:
      - Sample 0: err_f = 0, err_b = 0 -> Tie
      - Sample 1: err_f = 2, err_b = 0 -> Harm (err_f > err_b)
      - Sample 2: err_f = 1, err_b = 2 -> Help (err_f < err_b)
      - Sample 3: err_f = 0, err_b = 1 -> Help (err_f < err_b)
    Expected proportions:
      harm_rate = 1/4 = 0.25
      tie_rate  = 1/4 = 0.25
      help_rate = 2/4 = 0.50
    """
    yt = np.array([2, 1, 3, 0])
    yf = np.array([2, 3, 2, 0])
    yb = np.array([2, 1, 1, 1])

    res = compute_harm_rate(yt, yf, yb)
    assert pytest.approx(res["harm_rate"], rel=1e-5) == 0.25
    assert pytest.approx(res["tie_rate"], rel=1e-5) == 0.25
    assert pytest.approx(res["help_rate"], rel=1e-5) == 0.50
    assert pytest.approx(res["harm_rate"] + res["tie_rate"] + res["help_rate"], rel=1e-5) == 1.0


def test_audc_hand_calc():
    """
    5-step degradation curve example (levels 0 to 5):
    curve = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    Individual trapezoid areas:
      (1.0 + 0.8)/2 = 0.9
      (0.8 + 0.6)/2 = 0.7
      (0.6 + 0.4)/2 = 0.5
      (0.4 + 0.2)/2 = 0.3
      (0.2 + 0.0)/2 = 0.1
    Total area = 2.5
    Normalized by 5.0 -> Expected AUDC = 0.5
    """
    curve = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    audc = compute_audc(curve)
    assert pytest.approx(audc, rel=1e-5) == 0.5


def test_fdi_hand_calc():
    """
    Hand-calculated FDI example:
    f_clean = [[3.0, 4.0]] (norm = 5.0)
    f_deg   = [[6.0, 8.0]] (difference = [3.0, 4.0], difference norm = 5.0)
    Expected FDI = 5.0 / 5.0 = 1.0
    """
    f_clean = np.array([[3.0, 4.0]])
    f_deg   = np.array([[6.0, 8.0]])
    res = compute_fdi(f_clean, f_deg)
    assert pytest.approx(res["mean_fdi"], rel=1e-5) == 1.0


def test_pfr_hand_calc():
    """
    Hand-calculated Prediction Flip Rate example:
    y_clean = [0, 1, 2, 3]
    y_deg   = [0, 2, 2, 4]
    Flips occur at samples 1 and 3 (2 / 4 = 0.5)
    """
    yc = np.array([0, 1, 2, 3])
    yd = np.array([0, 2, 2, 4])
    pfr = compute_pfr(yc, yd)
    assert pytest.approx(pfr, rel=1e-5) == 0.5
