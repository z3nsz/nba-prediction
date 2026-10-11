"""Tests for the XGBoost model (src/xgboost_model.py)."""

import numpy as np

from src.split import split_by_season
from src.xgboost_model import REASONED_SETTINGS, choose_number_of_trees, predict_home_win, train_xgboost


def test_early_stopping_never_uses_validation_or_test(features):
    """Flipping every validation/test result must not change the chosen number of trees."""
    splits = split_by_season(features)
    flipped = features.copy()
    later = ~flipped["season"].isin(splits["train"]["season"].unique())
    flipped.loc[later, "home_team_win"] = 1 - flipped.loc[later, "home_team_win"]
    assert (choose_number_of_trees(splits["train"], REASONED_SETTINGS)
            == choose_number_of_trees(split_by_season(flipped)["train"], REASONED_SETTINGS))


def test_xgboost_is_reproducible_and_outputs_probabilities(features):
    splits = split_by_season(features)
    first = predict_home_win(train_xgboost(splits["train"], REASONED_SETTINGS, n_trees=50), splits["validation"])
    second = predict_home_win(train_xgboost(splits["train"], REASONED_SETTINGS, n_trees=50), splits["validation"])
    np.testing.assert_allclose(first, second, rtol=1e-6)
    assert ((first > 0) & (first < 1)).all()
