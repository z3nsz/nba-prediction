"""Tests for calibration and bootstrap helpers (src/compare_models.py)."""

import numpy as np
import pandas as pd
import pytest

from src.compare_models import bootstrap_log_loss_gap, calibration_bins, expected_calibration_error


def test_calibration_error_is_small_when_outcomes_follow_the_probabilities():
    """Simulated coin flips drawn from known probabilities should look well calibrated."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.2, 0.8, 50_000)
    outcomes = pd.Series((rng.uniform(size=probs.size) < probs).astype(int))
    assert expected_calibration_error(outcomes, probs) < 0.01


def test_calibration_error_catches_overconfidence():
    """Same outcomes, but the model claims 0.3 more than the truth for every game."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.2, 0.6, 50_000)
    outcomes = pd.Series((rng.uniform(size=probs.size) < probs).astype(int))
    assert expected_calibration_error(outcomes, probs + 0.3) == pytest.approx(0.3, abs=0.02)


def test_calibration_bins_have_equal_sizes():
    bins = calibration_bins(pd.Series([0, 1] * 50), np.linspace(0.1, 0.9, 100), n_bins=10)
    assert (bins["games"] == 10).all()


def test_bootstrap_gap_is_zero_for_identical_models():
    outcomes = pd.Series([1, 0, 1, 1, 0, 1])
    probs = np.array([0.7, 0.4, 0.6, 0.8, 0.3, 0.55])
    gap, low, high = bootstrap_log_loss_gap(outcomes, probs, probs, n_draws=200)
    assert gap == low == high == 0
