"""Replay past days of 2025-26 as if they were live, and compare with the original pipeline.

For a past date D: history = every game through 2024-25; current season = ESPN games from opening night to D,
with D's games stripped of their scores (they are "upcoming"). The live features for D's games must equal the
features the original stats.nba.com pipeline computed for the same games.

Usage:
    python -m src.live.replay
"""

import numpy as np
import pandas as pd

from src.feature_engineering import load_features_table
from src.live.team_features import TEAM_INPUTS, espn_season_games, load_history, team_features
from src.model_data import make_model_inputs

SEASON, OPENING_NIGHT = "2025-26", pd.Timestamp("2025-10-21")
REPLAY_DATES = ["2025-10-21", "2025-10-23", "2025-11-14", "2025-12-25", "2025-12-26", "2026-01-15",
                "2026-02-05", "2026-03-10", "2026-04-01", "2026-04-12"]


def replay_day(day: pd.Timestamp, history: pd.DataFrame) -> pd.DataFrame:
    season = espn_season_games(SEASON, OPENING_NIGHT, day, today=pd.Timestamp.max)
    on_day = season["game_date"] == day
    season.loc[on_day, ["home_score", "away_score", "home_team_win"]] = np.nan   # pretend they have not been played
    return team_features(history[history["season"] < SEASON], season).merge(
        season[["game_id", "game_date", "home_team", "away_team"]], on="game_id")


def compare(live: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Match live rows (ESPN ids) to original rows (NBA ids) by date and teams, then subtract."""
    original = original.assign(pair=[frozenset(p) for p in zip(original["home_team"], original["away_team"])])
    live = live.assign(pair=[frozenset(p) for p in zip(live["home_team"], live["away_team"])])
    merged = live.merge(original, on=["game_date", "pair"], suffixes=("_live", "_original"))
    flipped = merged["home_team_live"] != merged["home_team_original"]
    for name in TEAM_INPUTS:   # if ESPN lists the teams the other way round (neutral site), flip the sign
        sign = np.where(flipped & (name != "home_court"), -1, 1)
        merged[f"{name}_diff"] = merged[f"{name}_live"] * sign - merged[f"{name}_original"]
    return merged


if __name__ == "__main__":
    history = load_history()
    features = load_features_table()
    inputs = make_model_inputs(features).assign(game_date=features["game_date"], home_team=features["home_team"],
                                                away_team=features["away_team"])
    original = inputs[features["season"] == SEASON]
    results = []
    for date in REPLAY_DATES:
        day = pd.Timestamp(date)
        merged = compare(replay_day(day, history), original[original["game_date"] == day])
        worst = max(merged[f"{name}_diff"].abs().max() for name in TEAM_INPUTS)
        results.append({"date": date, "games": len(merged), "max_difference_any_input": worst})
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    print(f"\n{table['games'].sum()} games replayed; largest difference in any team input: "
          f"{table['max_difference_any_input'].max():.2e}")
