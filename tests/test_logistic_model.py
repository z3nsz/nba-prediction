"""Tests for the logistic regression model (src/logistic_model.py)."""

import numpy as np
import pytest

from src.logistic_model import predict_home_win, train_logistic
from src.model_data import FEATURE_SETS, make_model_inputs
from src.split import split_by_season

COLUMNS = FEATURE_SETS["elo + form + rest"]


@pytest.fixture(scope="module")
def model_and_splits(features):
    splits = split_by_season(features)
    return train_logistic(splits["train"], COLUMNS), splits


def test_equal_teams_at_neutral_site_are_a_coin_flip(model_and_splits):
    model, splits = model_and_splits
    equal_teams = make_model_inputs(splits["validation"]).iloc[:2][COLUMNS] * 0
    equal_teams["home_court"] = [1, 0]
    at_home, at_neutral = model.predict_proba(equal_teams)[:, 1]
    assert at_neutral == pytest.approx(0.5)
    assert 0.5 < at_home < 0.65, "home-court edge should be modest"


def test_swapping_teams_at_neutral_site_mirrors_the_probability(model_and_splits):
    model, splits = model_and_splits
    games = make_model_inputs(splits["validation"]).iloc[:50][COLUMNS].assign(home_court=0)
    swapped = (games * -1).assign(home_court=0)
    np.testing.assert_allclose(model.predict_proba(games)[:, 1], 1 - model.predict_proba(swapped)[:, 1])


def test_model_never_sees_validation_or_test_outcomes(features):
    """Changing every validation/test result must not change the trained model."""
    splits = split_by_season(features)
    flipped = features.copy()
    later = ~flipped["season"].isin(splits["train"]["season"].unique())
    flipped.loc[later, "home_team_win"] = 1 - flipped.loc[later, "home_team_win"]
    original = train_logistic(splits["train"], COLUMNS)
    retrained = train_logistic(split_by_season(flipped)["train"], COLUMNS)
    validation = splits["validation"]
    np.testing.assert_allclose(predict_home_win(original, validation, COLUMNS),
                               predict_home_win(retrained, validation, COLUMNS))
