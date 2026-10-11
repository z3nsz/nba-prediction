"""Tests for the scoring rules and naive baselines (src/evaluate.py)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_baselines, score_predictions
from src.split import split_by_season


def test_scores_on_a_hand_checked_example():
    outcomes = pd.Series([1, 0, 1, 0])
    probs = np.array([0.8, 0.3, 0.4, 0.5])
    scores = score_predictions(outcomes, probs)
    # Picks: right, right, wrong, and a 50/50 that earns half credit -> 2.5 / 4.
    assert scores["accuracy"] == pytest.approx(0.625)
    # Brier: mean of 0.2^2, 0.3^2, 0.6^2, 0.5^2.
    assert scores["brier"] == pytest.approx((0.04 + 0.09 + 0.36 + 0.25) / 4)
    # Log loss: mean of -log(probability given to what actually happened).
    assert scores["log_loss"] == pytest.approx(-np.mean(np.log([0.8, 0.7, 0.4, 0.5])))


def test_coin_flip_scores_are_the_known_constants():
    outcomes = pd.Series([1, 0, 1, 1, 0])
    scores = score_predictions(outcomes, np.full(5, 0.5))
    assert scores["accuracy"] == 0.5
    assert scores["brier"] == pytest.approx(0.25)
    assert scores["log_loss"] == pytest.approx(np.log(2))
    assert scores["auc"] == 0.5


def test_baselines_never_touch_the_test_season(features):
    results = evaluate_baselines(split_by_season(features))
    assert "test" not in results.index.get_level_values("split")
