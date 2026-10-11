"""Leakage and correctness tests for ratings and strength of schedule (src/strength_features.py)."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import ELO_HOME_ADVANTAGE, compute_elo
from src.strength_features import ELO_POINTS_PER_POINT_OF_MARGIN, strength_features, team_box_scores


def hand_made_games() -> pd.DataFrame:
    """Three games between two teams with easy numbers (100 possessions for each team every game)."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "game_date": pd.to_datetime(["2024-11-01", "2024-11-03", "2024-11-05"]),
        "season": ["2024-25"] * 3,
        "home_team": ["AAA", "BBB", "AAA"],
        "away_team": ["BBB", "AAA", "BBB"],
        "home_score": [110, 100, 120],
        "away_score": [100, 90, 80],
        "home_team_win": [1, 1, 1],
        "is_neutral_site": [0, 0, 0],
        # FGA + 0.44*FTA - OREB + TOV = 80 + 0.44*25 - 5 + 14 = 100
        "home_fga": [80] * 3, "home_fta": [25] * 3, "home_oreb": [5] * 3, "home_tov": [14] * 3,
        "away_fga": [80] * 3, "away_fta": [25] * 3, "away_oreb": [5] * 3, "away_tov": [14] * 3,
    })


def test_ratings_on_hand_made_games():
    games = hand_made_games()
    result = strength_features(games, compute_elo(games), ELO_HOME_ADVANTAGE).set_index(["game_id", "team"])
    # Before g3, AAA scored 110 + 90 = 200 and allowed 100 + 100 = 200 over 200 possessions.
    assert result.loc[("g3", "AAA"), "off_rating_last10"] == pytest.approx(100.0)
    assert result.loc[("g3", "AAA"), "def_rating_last10"] == pytest.approx(100.0)
    # Before g2, AAA had one game: 110 scored, 100 allowed per 100 possessions -> net +10.
    assert result.loc[("g2", "AAA"), "net_rating_last5"] == pytest.approx(10.0)
    # Season opener: no history.
    assert result.loc[("g1", "AAA")].isna().all()


def test_expected_margin_uses_home_advantage_for_home_and_against_away():
    games = hand_made_games().iloc[[0]]
    box = team_box_scores(games, compute_elo(games), ELO_HOME_ADVANTAGE).set_index("team")
    # Both teams start at 1500, so only the home bonus matters: +100 / 28 for home, -100 / 28 for away.
    assert box.loc["AAA", "expected_margin"] == pytest.approx(ELO_HOME_ADVANTAGE / ELO_POINTS_PER_POINT_OF_MARGIN)
    assert box.loc["BBB", "expected_margin"] == pytest.approx(-ELO_HOME_ADVANTAGE / ELO_POINTS_PER_POINT_OF_MARGIN)


def test_strength_features_match_brute_force(games):
    """For random team-games, recompute from scratch using only games on strictly earlier dates."""
    dev_games = games[games["season"] != "2025-26"]  # never needed here; keeps the test off the test season
    elo = compute_elo(dev_games)
    features = strength_features(dev_games, elo, ELO_HOME_ADVANTAGE)
    box = team_box_scores(dev_games, elo, ELO_HOME_ADVANTAGE)
    for _, row in features.merge(box[["game_id", "team", "game_date", "season"]]).sample(150, random_state=0).iterrows():
        past = box[(box["team"] == row["team"]) & (box["season"] == row["season"])
                   & (box["game_date"] < row["game_date"])].tail(10)
        if past.empty:
            assert np.isnan(row["off_rating_last10"])
            continue
        assert row["off_rating_last10"] == pytest.approx(100 * past["points_scored"].sum() / past["possessions"].sum())
        assert row["def_rating_last10"] == pytest.approx(100 * past["points_allowed"].sum() / past["possessions"].sum())
        assert row["opp_elo_last10"] == pytest.approx(past["opp_elo"].mean())
        assert row["margin_vs_elo_last10"] == pytest.approx(past["margin_vs_elo"].mean())


def test_net_rating_difference_is_exactly_offense_plus_defense(features):
    """Documents why the ratings group leaves net rating out (it would duplicate information)."""
    from src.model_data import make_model_inputs
    inputs = make_model_inputs(features[features["season"] != "2025-26"])
    for window in [5, 10]:
        np.testing.assert_allclose(inputs[f"net_rating_last{window}_diff"],
                                   inputs[f"off_rating_last{window}_diff"] + inputs[f"def_rating_last{window}_diff"],
                                   atol=1e-9)
