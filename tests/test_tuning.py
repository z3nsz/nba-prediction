"""Tests for the tuning helpers (src/tuning.py)."""

import pandas as pd

from src.tuning import best_settings, grid_combinations


def test_grid_combinations_lists_every_pair():
    combos = grid_combinations({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert {"a": 2, "b": "z"} in combos


def test_best_settings_restores_plain_types():
    """pandas turns 7 into 7.0 and None into NaN when a column mixes them; best_settings undoes that."""
    results = pd.DataFrame({"max_features": [7, 3], "max_depth": [None, 4], "learning_rate": [0.1, 0.3],
                            "validation_log_loss": [0.61, 0.62]})
    best = best_settings(results)
    assert best == {"max_features": 7, "max_depth": None, "learning_rate": 0.1}
    assert type(best["max_features"]) is int
