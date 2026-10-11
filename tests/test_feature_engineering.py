"""Leakage and correctness tests for pre-game features (src/feature_engineering.py)."""

import numpy as np
import pandas as pd
import pytest

from src.collect_data import SEASONS
from src.feature_engineering import (
    ELO_START,
    GAME_COLUMNS,
    ROLLING_FEATURES,
    add_rest_features,
    build_features_table,
    compute_elo,
    previous_games_mean,
)
from src.strength_features import STRENGTH_FEATURES
from src.travel_features import TRAVEL_FEATURES

ROLLING_NAMES = [name for name, _, _ in ROLLING_FEATURES]
PRE_GAME_COLUMNS = (
    [f"{side}_{name}" for side in ["home", "away"] for name in ROLLING_NAMES + STRENGTH_FEATURES + TRAVEL_FEATURES]
    + ["home_elo", "away_elo", "elo_home_win_prob", "elo_diff"]
)


# ---------- Unit tests on small hand-made tables ----------

def test_previous_games_mean_excludes_current_game():
    wins = pd.Series([1, 0, 1, 1, 0])
    result = previous_games_mean(wins, window=3)
    # Game 1 has no history; game 4 averages games 1-3 (1, 0, 1), NOT games 2-4.
    expected = [np.nan, 1.0, 0.5, 2 / 3, 2 / 3]
    np.testing.assert_allclose(result, expected)


def test_rest_features_on_example_schedule():
    schedule = pd.DataFrame({
        "team": ["AAA"] * 4,
        "season": ["2024-25"] * 4,
        "game_date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04", "2025-01-08"]),
    })
    result = add_rest_features(schedule)
    np.testing.assert_array_equal(result["days_since_last_game"], [np.nan, 2, 1, 4])
    assert result["is_back_to_back"].tolist() == [0, 0, 1, 0]


def test_elo_uses_ratings_from_before_each_game():
    two_games = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": pd.to_datetime(["2024-11-01", "2024-11-03"]),
        "season": ["2024-25", "2024-25"],
        "home_team": ["AAA", "AAA"],
        "away_team": ["BBB", "BBB"],
        "home_score": [110, 100],
        "away_score": [100, 105],
        "home_team_win": [1, 0],
        "is_neutral_site": [0, 0],
    })
    elo = compute_elo(two_games).set_index("game_id")
    # First meeting: nobody has played yet, so both start at the default rating.
    assert elo.loc["g1", "home_elo"] == elo.loc["g1", "away_elo"] == ELO_START
    # Second meeting reflects only game 1: the winner went up and the loser went down by the same amount.
    assert elo.loc["g2", "home_elo"] > ELO_START > elo.loc["g2", "away_elo"]
    assert elo.loc["g2", "home_elo"] - ELO_START == pytest.approx(ELO_START - elo.loc["g2", "away_elo"])


# ---------- Tests on the real dataset ----------

def test_features_contain_only_modeling_seasons(features, games):
    assert set(features["season"]) == set(SEASONS)
    assert len(features) == games["season"].isin(SEASONS).sum()
    assert features["game_id"].is_unique


def test_no_same_game_results_in_features(features, games):
    """Box-score stats from the game itself are only known after it ends.

    Every games-table column that is not deliberately copied over (GAME_COLUMNS) is a
    same-game stat or ID, so none of them may appear in the features table.
    """
    forbidden = set(games.columns) - set(GAME_COLUMNS) - {"home_team_id", "away_team_id"}
    assert {"home_score", "home_fga", "away_oreb"} <= forbidden  # the check covers new box-score columns
    assert not forbidden & set(features.columns)


def test_rolling_features_match_brute_force(features, team_games):
    """Recompute features for random games using ONLY games on strictly earlier dates."""
    sample = features.sample(200, random_state=0)
    for _, game in sample.iterrows():
        for side in ["home", "away"]:
            past = team_games[
                (team_games["team"] == game[f"{side}_team"])
                & (team_games["season"] == game["season"])
                & (team_games["game_date"] < game["game_date"])
            ].sort_values("game_date")
            for name, source_column, window in ROLLING_FEATURES:
                expected = past[source_column].tail(window).mean()
                actual = game[f"{side}_{name}"]
                assert (pd.isna(expected) and pd.isna(actual)) or np.isclose(expected, actual), (
                    f"{game['game_id']} {side}_{name}: expected {expected}, got {actual}"
                )


def test_changing_a_result_does_not_change_that_games_features(games, features):
    """Poison test: flip one game's result. Its own pre-game features must not move."""
    target_id = features["game_id"].iloc[4000]
    poisoned = games.copy()
    row = poisoned.index[poisoned["game_id"] == target_id][0]
    poisoned.loc[row, ["home_score", "away_score"]] = poisoned.loc[row, ["away_score", "home_score"]].values
    poisoned.loc[row, "home_team_win"] = 1 - poisoned.loc[row, "home_team_win"]

    rebuilt = build_features_table(poisoned)
    before = features.set_index("game_id")[PRE_GAME_COLUMNS]
    after = rebuilt.set_index("game_id")[PRE_GAME_COLUMNS]
    pd.testing.assert_series_equal(before.loc[target_id], after.loc[target_id])

    # Sanity check that the test can fail: later games SHOULD change.
    later = features.loc[features["game_date"] > features.loc[features["game_id"] == target_id, "game_date"].iloc[0], "game_id"]
    assert not before.loc[later].equals(after.loc[later])


def test_missing_rolling_features_only_for_season_openers(features):
    for side in ["home", "away"]:
        has_missing = features[[f"{side}_{name}" for name in ROLLING_NAMES + STRENGTH_FEATURES]].isna().any(axis=1)
        is_opener = features[f"{side}_games_played_before"] == 0
        assert (has_missing == is_opener).all()


def test_every_team_has_exactly_one_opener_per_season(features):
    openers = pd.concat([
        features.loc[features["home_games_played_before"] == 0, ["season", "home_team"]].set_axis(["season", "team"], axis=1),
        features.loc[features["away_games_played_before"] == 0, ["season", "away_team"]].set_axis(["season", "team"], axis=1),
    ])
    assert len(openers) == 30 * len(SEASONS)
    assert not openers.duplicated().any()


def test_feature_values_in_valid_ranges(features):
    for side in ["home", "away"]:
        for window in [5, 10]:
            assert features[f"{side}_win_pct_last{window}"].dropna().between(0, 1).all()
        assert features[f"{side}_days_since_last_game"].dropna().ge(1).all()
        assert features[f"{side}_is_back_to_back"].isin([0, 1]).all()
    assert features["elo_home_win_prob"].between(0, 1).all()
    assert features["home_court"].isin([0, 1]).all()


def test_feature_building_is_reproducible(games, features):
    pd.testing.assert_frame_equal(build_features_table(games), features)


def test_elo_home_advantage_setting_is_applied():
    """With no home advantage, two unrated teams are a 50/50 game; with +100 the home side is favored."""
    one_game = pd.DataFrame({
        "game_id": ["g1"], "game_date": pd.to_datetime(["2024-11-01"]), "season": ["2024-25"],
        "home_team": ["AAA"], "away_team": ["BBB"], "home_score": [100], "away_score": [90],
        "home_team_win": [1], "is_neutral_site": [0],
    })
    assert compute_elo(one_game, home_advantage=0)["elo_home_win_prob"].iloc[0] == pytest.approx(0.5)
    assert compute_elo(one_game, home_advantage=100)["elo_home_win_prob"].iloc[0] == pytest.approx(0.640, abs=0.001)
