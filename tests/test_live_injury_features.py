"""Tests for live injury inputs (src/live/injury_features.py) on hand-made data."""

import numpy as np
import pandas as pd
import pytest

import src.live.injury_features as lf


@pytest.mark.parametrize("report, espn", [("Portis Jr., Bobby", "Bobby Portis"), ("Butler III, Jimmy", "Jimmy Butler"),
                                          ("Doncic, Luka", "Luka Dončić"), ("Payton II, Gary", "Gary Payton II")])
def test_live_key_ignores_suffixes_and_accents(report, espn):
    assert lf.live_key(report) == lf.live_key(espn)


def test_history_rows_reproduce_the_saved_totals():
    value_state = pd.DataFrame({"name_key": [lf.LEAGUE_KEY, "star"], "gs_sum": [1000.0, 300.0],
                                "min_sum": [3000.0, 600.0], "season": "2025-26"})
    rows = lf.history_logs(value_state, pd.Timestamp("2026-10-20"))
    assert (lf.player_game_score(rows) == rows["PTS"]).all(), "only PTS is non-zero, so Game Score = PTS"
    assert rows.loc[rows["name_key"] == "star", "season"].iloc[0] == "2025-26"
    assert rows.loc[rows["name_key"] == lf.LEAGUE_KEY, "season"].iloc[0] == lf.HISTORY_SEASON


def test_history_listings_expand_back_to_the_counts():
    state = pd.DataFrame({"name_key": ["a", "a"], "status": ["Questionable", "Out"], "plays": [3, 0], "listings": [5, 2]})
    rows = lf.history_listings(state, pd.Timestamp("2026-10-20"))
    counts = rows.groupby("status")["played"].agg(["sum", "size"])
    assert counts.loc["Questionable"].tolist() == [3, 5] and counts.loc["Out"].tolist() == [0, 2]


def test_injury_diffs_align_by_team_not_by_row_label():
    """Regression test: games with non-default row labels (e.g. 8303) once produced all-NaN -> 0 values."""
    games = pd.DataFrame({"game_id": ["g1"], "home_team": ["IND"], "away_team": ["OKC"]}, index=[8303])
    per_team = pd.DataFrame({"game_id": ["g1"], "team_abbr": ["OKC"], "exp_missing_minutes": [75.0],
                             "starters_out": [2], "exp_missing_value": [-1.0], "exp_stars_missing": [0.0]})
    known = pd.DataFrame({"game_id": ["g1", "g1"], "team_abbr": ["IND", "OKC"], "known": [True, True]})
    out = lf.injury_diffs(games, per_team, known).iloc[0]
    assert out["away_exp_missing_minutes"] == 75.0 and out["home_exp_missing_minutes"] == 0.0
    assert out["exp_missing_minutes_diff"] == 75.0 and out["starters_out_diff"] == 2


def test_unknown_report_gives_zero_difference():
    games = pd.DataFrame({"game_id": ["g1"], "home_team": ["IND"], "away_team": ["OKC"]})
    per_team = pd.DataFrame({"game_id": ["g1"], "team_abbr": ["OKC"], "exp_missing_minutes": [75.0],
                             "starters_out": [2], "exp_missing_value": [-1.0], "exp_stars_missing": [0.0]})
    known = pd.DataFrame({"game_id": ["g1", "g1"], "team_abbr": ["IND", "OKC"], "known": [True, False]})
    out = lf.injury_diffs(games, per_team, known).iloc[0]
    assert np.isnan(out["away_exp_missing_minutes"]) and out["exp_missing_minutes_diff"] == 0
