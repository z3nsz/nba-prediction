"""Tests for walk-forward evaluation (src/walk_forward.py)."""

import json

import numpy as np
import pandas as pd

from src.tuning import TUNED_SETTINGS_PATH
from src.walk_forward import FOLDS, TEST_SEASONS, date_block_bootstrap, walk_forward_predictions


def test_folds_always_train_on_the_past_and_never_touch_the_test_season():
    for train_seasons, eval_season in FOLDS:
        assert all(season < eval_season for season in train_seasons)
        assert eval_season not in TEST_SEASONS
        assert not set(train_seasons) & set(TEST_SEASONS)


def test_test_season_results_cannot_affect_walk_forward_predictions(features):
    """Flip every 2025-26 result: walk-forward predictions must be identical."""
    tuned = json.loads(TUNED_SETTINGS_PATH.read_text())
    runs = [("logistic", "phase 2 (7 inputs)")]
    flipped = features.copy()
    in_test = flipped["season"].isin(TEST_SEASONS)
    flipped.loc[in_test, "home_team_win"] = 1 - flipped.loc[in_test, "home_team_win"]
    original = walk_forward_predictions(features, runs, tuned)
    after = walk_forward_predictions(flipped, runs, tuned)
    assert not original["season"].isin(TEST_SEASONS).any()
    pd.testing.assert_frame_equal(original, after)


def test_block_bootstrap_is_zero_for_identical_predictions():
    predictions = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]),
        "home_team_win": [1, 0, 1, 0],
        "a": [0.6, 0.4, 0.7, 0.3],
    })
    predictions["b"] = predictions["a"]
    gap, low, high = date_block_bootstrap(predictions, "a", "b", n_draws=100)
    assert gap == low == high == 0


def test_block_bootstrap_detects_a_clearly_worse_model():
    rng = np.random.default_rng(0)
    n = 2000
    probs = rng.uniform(0.2, 0.8, n)
    predictions = pd.DataFrame({
        "game_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D"),
        "home_team_win": (rng.uniform(size=n) < probs).astype(int),
        "honest": probs,
        "coin_flip": 0.5,
    })
    gap, low, high = date_block_bootstrap(predictions, "coin_flip", "honest", n_draws=300)
    assert low > 0, "a coin flip should be clearly worse than the true probabilities"
