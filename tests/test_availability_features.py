"""Leakage tests for player-availability features (src/availability_features.py)."""

import pandas as pd
import pytest

from src.availability_features import attach_expected_minutes, expected_minutes_table


def player_logs() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2024-25"] * 3,
        "PLAYER_ID": [1, 1, 1],
        "name_key": ["starplayer"] * 3,
        "GAME_DATE": pd.to_datetime(["2024-11-01", "2024-11-03", "2024-11-05"]),
        "MIN": [30, 40, 20],
        # Only points are non-zero, so Game Score equals points: 10, 30, 50.
        "PTS": [10, 30, 50], "FGM": [0] * 3, "FGA": [0] * 3, "FTA": [0] * 3, "FTM": [0] * 3, "OREB": [0] * 3,
        "DREB": [0] * 3, "STL": [0] * 3, "AST": [0] * 3, "BLK": [0] * 3, "PF": [0] * 3, "TOV": [0] * 3,
        "PLUS_MINUS": [5, -5, 15],
    })


def test_expected_minutes_use_only_earlier_dates():
    minutes = expected_minutes_table(player_logs())
    listed = pd.DataFrame({"season": ["2024-25"] * 4, "name_key": ["starplayer"] * 4,
                           "game_date": pd.to_datetime(["2024-11-01", "2024-11-03", "2024-11-04", "2024-11-05"])})
    result = attach_expected_minutes(listed, minutes).set_index("game_date")["avg_minutes"]
    assert pd.isna(result["2024-11-01"]), "no earlier games: unknown, not that day's 30 minutes"
    assert result["2024-11-03"] == 30, "only the Nov 1 game counts, not Nov 3's own 40 minutes"
    assert result["2024-11-04"] == 35
    assert result["2024-11-05"] == 35, "the Nov 5 game itself (20 minutes) must not count"
    impact = attach_expected_minutes(listed, minutes).set_index("game_date")
    assert impact.loc["2024-11-05", "avg_game_score"] == 20, "average of 10 and 30, not the Nov 5 game's 50"
    assert impact.loc["2024-11-05", "avg_plus_minus"] == 0


def test_expected_minutes_reset_each_season():
    logs = pd.concat([player_logs(), player_logs().assign(season="2025-26",
                     GAME_DATE=pd.to_datetime(["2025-11-01", "2025-11-03", "2025-11-05"]))])
    minutes = expected_minutes_table(logs)
    listed = pd.DataFrame({"season": ["2025-26"], "name_key": ["starplayer"],
                           "game_date": pd.to_datetime(["2025-11-01"])})
    assert pd.isna(attach_expected_minutes(listed, minutes)["avg_minutes"].iloc[0])


def test_built_availability_table_respects_the_rules():
    """Checks on the saved table: development seasons only, no invented values for unknown reports."""
    from src.availability_features import AVAILABILITY_PATH
    from src.clean_data import load_games_table
    from src.walk_forward import TEST_SEASONS
    if not AVAILABILITY_PATH.exists():
        pytest.skip("availability features not built; run python -m src.availability_features")
    table = pd.read_csv(AVAILABILITY_PATH, dtype={"game_id": str})
    seasons = table.merge(load_games_table()[["game_id", "season"]], on="game_id")["season"]
    assert not seasons.isin(TEST_SEASONS).any(), "no injury features may exist for the test season"
    for side in ["home", "away"]:
        unknown = table[f"{side}_report_found"] == 0
        assert table.loc[unknown, f"{side}_out_minutes"].isna().all(), "unknown must stay NaN, not 0"
        assert table.loc[~unknown, f"{side}_out_minutes"].notna().all()
        assert (table[f"{side}_out_minutes"].dropna() >= 0).all()
