"""Chronological train / validation / test split by season.

Games are split by season, never at random, so every model is always trained
on the past and evaluated on the future, just like a real forecast.

    train       2020-21, 2021-22, 2022-23   fit models and calibration here
    validation  2023-24, 2024-25            compare models and choose settings here
    test        2025-26                     evaluate ONCE, at the very end

Usage:
    python -m src.split
"""

import pandas as pd

from src.feature_engineering import load_features_table

SPLIT_SEASONS = {
    "train": ["2020-21", "2021-22", "2022-23"],
    "validation": ["2023-24", "2024-25"],
    "test": ["2025-26"],
}


def split_by_season(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {"train": ..., "validation": ..., "test": ...}, each sorted by date."""
    splits = {
        name: features[features["season"].isin(seasons)].sort_values(["game_date", "game_id"]).reset_index(drop=True)
        for name, seasons in SPLIT_SEASONS.items()
    }
    check_split(features, splits)
    return splits


def check_split(features: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the split loses games, overlaps, or lets the future leak into the past."""
    all_ids = [game_id for split in splits.values() for game_id in split["game_id"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("a game appears in more than one split")
    if set(all_ids) != set(features["game_id"]):
        raise ValueError("some games are not in any split")

    names = list(splits)
    for earlier, later in zip(names, names[1:]):
        if splits[earlier]["game_date"].max() >= splits[later]["game_date"].min():
            raise ValueError(f"{earlier} games overlap in time with {later} games")


def describe_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One summary row per split: size, date range, and home win rate."""
    return pd.DataFrame({
        name: {
            "seasons": ", ".join(SPLIT_SEASONS[name]),
            "games": len(split),
            "share": None,
            "first_game": split["game_date"].min().date(),
            "last_game": split["game_date"].max().date(),
            "home_win_rate": round(split["home_team_win"].mean(), 3),
        }
        for name, split in splits.items()
    }).T.assign(share=lambda table: (table["games"] / table["games"].sum()).map("{:.0%}".format))


if __name__ == "__main__":
    splits = split_by_season(load_features_table())
    print(describe_splits(splits).to_string())
