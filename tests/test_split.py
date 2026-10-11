"""Tests for the chronological train / validation / test split (src/split.py)."""

import pandas as pd
import pytest

from src.collect_data import SEASONS
from src.split import SPLIT_SEASONS, check_split, split_by_season


def test_split_covers_every_modeling_season_once():
    listed = [season for seasons in SPLIT_SEASONS.values() for season in seasons]
    assert sorted(listed) == sorted(SEASONS)


def test_splits_are_chronological_and_complete(features):
    splits = split_by_season(features)
    assert sum(len(split) for split in splits.values()) == len(features)
    assert splits["train"]["game_date"].max() < splits["validation"]["game_date"].min()
    assert splits["validation"]["game_date"].max() < splits["test"]["game_date"].min()


def test_check_split_rejects_overlapping_dates():
    """Hand-made example: a 'train' game played after a 'validation' game must be rejected."""
    games = pd.DataFrame({
        "game_id": ["a", "b"],
        "game_date": pd.to_datetime(["2024-03-01", "2024-02-01"]),
    })
    bad_splits = {"train": games.iloc[[0]], "validation": games.iloc[[1]]}
    with pytest.raises(ValueError, match="overlap in time"):
        check_split(games, bad_splits)
