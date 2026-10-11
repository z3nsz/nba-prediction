"""Tests for the random forest model (src/random_forest_model.py)."""

import numpy as np

from src.random_forest_model import REASONED_SETTINGS, predict_home_win, train_random_forest
from src.split import split_by_season

SMALL = {**REASONED_SETTINGS, "n_estimators": 50}  # fewer trees keeps the test fast


def test_random_forest_is_reproducible_and_outputs_probabilities(features):
    splits = split_by_season(features)
    first = predict_home_win(train_random_forest(splits["train"], SMALL), splits["validation"])
    second = predict_home_win(train_random_forest(splits["train"], SMALL), splits["validation"])
    # Parallel training can change the order trees are summed in, which moves the
    # 16th decimal place; anything beyond that tiny tolerance is a real difference.
    np.testing.assert_allclose(first, second, rtol=1e-12)
    assert ((first > 0) & (first < 1)).all(), "min_samples_leaf should prevent 0% / 100% predictions"
