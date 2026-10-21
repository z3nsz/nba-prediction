"""Tests for the saved live model (src/live/model.py)."""

import numpy as np
import pandas as pd
import pytest

from src.live.model import MODEL_PATH, load_model, predict


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("live model not built; run python -m src.live.model")
    return load_model()


def test_model_has_the_locked_inputs_and_no_intercept(model):
    assert model["inputs"][:2] == ["elo_diff", "home_court"] and len(model["inputs"]) == 11
    assert model["trained_on"] == ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    assert "intercept" not in model


def test_equal_teams_at_a_neutral_site_are_a_coin_flip(model):
    zeros = pd.DataFrame([{name: 0.0 for name in model["inputs"]}])
    assert predict(model, zeros)[0] == pytest.approx(0.5)


def test_swapping_home_and_away_at_a_neutral_site_mirrors_the_probability(model):
    rng = np.random.default_rng(0)
    games = pd.DataFrame(rng.normal(size=(20, len(model["inputs"]))), columns=model["inputs"]).assign(home_court=0)
    swapped = (-games).assign(home_court=0)
    np.testing.assert_allclose(predict(model, games), 1 - predict(model, swapped))


def test_home_court_alone_gives_a_modest_home_edge(model):
    home = pd.DataFrame([{name: 0.0 for name in model["inputs"]}]).assign(home_court=1)
    assert 0.52 < predict(model, home)[0] < 0.60
