"""Tests for live team features (src/live/team_features.py)."""

import pandas as pd
import pytest

import src.live.team_features as tf


def board(rows):
    base = {"season_type": 2, "neutral_site": False, "note": "", "tip_off_utc": pd.Timestamp("2026-01-01T00:00Z")}
    return pd.DataFrame([{**base, **row} for row in rows])


def test_skips_all_star_cup_final_and_postponed_games(monkeypatch):
    days = {
        "2026-01-01": board([
            {"espn_id": "1", "home_team": "BOS", "away_team": "NYK", "home_score": 100, "away_score": 90, "completed": True, "status": "Final"},
            {"espn_id": "2", "home_team": "CHI", "away_team": "MIA", "home_score": 0, "away_score": 0, "completed": False, "status": "Postponed"},
            {"espn_id": "3", "home_team": "STARS", "away_team": "WORLD", "home_score": 40, "away_score": 38, "completed": True, "status": "Final"},
            {"espn_id": "4", "home_team": "NYK", "away_team": "SAS", "home_score": 124, "away_score": 113, "completed": True,
             "status": "Final", "note": "NBA Cup Championship"},
        ]),
        "2026-01-02": board([
            {"espn_id": "5", "home_team": "LAL", "away_team": "GSW", "home_score": 0, "away_score": 0, "completed": False, "status": "Scheduled"},
            {"espn_id": "6", "home_team": "DEN", "away_team": "PHX", "home_score": 0, "away_score": 0, "completed": False, "status": "Postponed"},
        ]),
    }
    monkeypatch.setattr(tf, "scoreboard", lambda day, cache=True: days[f"{day:%Y-%m-%d}"])
    games = tf.espn_season_games("2025-26", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-02"))
    assert list(games["game_id"]) == ["espn-1", "espn-5"]
    assert games.set_index("game_id").loc["espn-5", "home_score"] != games.set_index("game_id").loc["espn-5", "home_score"], \
        "the scheduled game has no score yet (NaN)"


@pytest.mark.parametrize("date", ["2025-12-25", "2026-02-05", "2026-04-12"])
def test_live_features_equal_the_training_pipeline_on_replayed_days(date):
    from src.feature_engineering import load_features_table
    from src.live.replay import SEASON, compare, replay_day
    from src.model_data import make_model_inputs
    if not tf.HISTORY_PATH.exists():
        pytest.skip("history not built; run python -m src.live.team_features")
    features = load_features_table()
    original = make_model_inputs(features).assign(
        game_date=features["game_date"], home_team=features["home_team"], away_team=features["away_team"])
    day = pd.Timestamp(date)
    merged = compare(replay_day(day, tf.load_history()), original[(features["season"] == SEASON) & (features["game_date"] == day)])
    assert len(merged) > 0
    for name in tf.TEAM_INPUTS:
        assert merged[f"{name}_diff"].abs().max() < 1e-9, name
