"""Shared test fixtures.

Fixtures build the datasets once per test run straight from data/raw/, so the
tests check the current code, not whatever CSVs happen to be on disk. If the
raw data has not been downloaded, data tests are skipped with a clear message
(unit tests on small hand-made tables still run).
"""

import pytest

from src.clean_data import build_games_table, load_all_raw_seasons
from src.collect_data import ALL_SEASONS, raw_file_path
from src.feature_engineering import build_features_table, to_team_games


@pytest.fixture(scope="session")
def raw():
    missing = [season for season in ALL_SEASONS if not raw_file_path(season).exists()]
    if missing:
        pytest.skip(f"raw data not downloaded for {missing}; run: python -m src.collect_data")
    return load_all_raw_seasons()


@pytest.fixture(scope="session")
def games(raw):
    return build_games_table(raw)


@pytest.fixture(scope="session")
def features(games):
    return build_features_table(games)


@pytest.fixture(scope="session")
def team_games(games):
    return to_team_games(games)
