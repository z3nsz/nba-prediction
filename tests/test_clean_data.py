"""Validation tests for the one-row-per-game table (src/clean_data.py)."""

import pandas as pd
import pytest

from src.clean_data import (
    KNOWN_NEUTRAL_SITE_GAME_IDS,
    add_home_away_flags,
    build_games_table,
    raw_data_quality_report,
)


def test_every_game_appears_exactly_once(games, raw):
    assert games["game_id"].is_unique
    assert set(games["game_id"]) == set(raw["GAME_ID"])


def test_every_raw_game_has_two_rows_and_two_distinct_teams(raw):
    per_game = raw.groupby("GAME_ID")
    assert (per_game.size() == 2).all()
    assert (per_game["TEAM_ID"].nunique() == 2).all()


def test_each_game_has_two_distinct_teams(games):
    assert (games["home_team"] != games["away_team"]).all()
    assert games[["home_team", "away_team"]].notna().all().all()


def test_home_team_matches_api_matchup(games, raw):
    """Wherever the API names a home team ("BOS vs. NYK"), our home_team must be that team."""
    flagged = add_home_away_flags(raw)
    api_home = flagged.loc[flagged["IS_HOME"]].set_index("GAME_ID")["TEAM_ABBREVIATION"]
    with_api_home = games[games["game_id"].isin(api_home.index)]
    assert (with_api_home["home_team"].values == api_home.loc[with_api_home["game_id"]].values).all()


def test_scores_are_valid(games):
    for column in ["home_score", "away_score"]:
        assert pd.api.types.is_integer_dtype(games[column])
        assert games[column].between(50, 200).all(), f"implausible values in {column}"
    assert (games["home_score"] != games["away_score"]).all(), "NBA games cannot end tied"


def test_target_is_binary_and_matches_scores(games):
    assert set(games["home_team_win"].unique()) <= {0, 1}
    expected = (games["home_score"] > games["away_score"]).astype(int)
    assert (games["home_team_win"] == expected).all()


def test_target_matches_api_win_loss_column(games, raw):
    winners = raw.loc[raw["WL"] == "W"].set_index("GAME_ID")["TEAM_ABBREVIATION"]
    our_winners = games["home_team"].where(games["home_team_win"] == 1, games["away_team"])
    assert (our_winners.values == winners.loc[games["game_id"]].values).all()


def test_known_neutral_site_ids_exist_and_are_flagged(games):
    flagged = games.set_index("game_id")["is_neutral_site"]
    for game_id in KNOWN_NEUTRAL_SITE_GAME_IDS:
        assert game_id in flagged.index, f"{game_id} listed as neutral-site but not in the data"
        assert flagged[game_id] == 1


def test_only_regular_season_games(raw):
    # Regular-season game IDs start with "002" (preseason "001", playoffs "004").
    assert raw["GAME_ID"].str.startswith("002").all()


def test_quality_report_counts_missing_values():
    """Hand-made example: the report must surface missing values, not hide them."""
    toy = pd.DataFrame({
        "GAME_ID": ["g1", "g1"],
        "TEAM_ID": [1, 2],
        "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
        "WL": ["W", "L"],
        "FT_PCT": [0.8, None],  # e.g. a team that attempted no free throws
    })
    report = raw_data_quality_report(toy)
    assert report["missing_values_by_column"] == {"FT_PCT": 1}
    assert report["games_without_exactly_one_winner"] == 0


def test_cleaning_is_reproducible(raw, games):
    rebuilt = build_games_table(raw)
    pd.testing.assert_frame_equal(rebuilt, games)


@pytest.mark.parametrize("column", ["home_score", "away_score", "home_team", "away_team", "game_date"])
def test_no_missing_values_in_core_columns(games, column):
    assert games[column].notna().all()
