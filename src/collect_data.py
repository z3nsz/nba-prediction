"""Download raw historical NBA team game logs with nba_api.

Each season is saved untouched to data/raw/team_game_logs_<season>.csv.
Seasons already on disk are skipped, so re-running the script does not
hit the API again unless --force is passed.

Usage:
    python -m src.collect_data            # download any missing seasons
    python -m src.collect_data --force    # re-download every season
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog

# Build paths from this file's location so the code works on any computer.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Seasons the model is trained and evaluated on.
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
# Earlier seasons used only to "warm up" Elo ratings so they are realistic by
# October 2020. They never become rows in the features file.
WARMUP_SEASONS = ["2018-19", "2019-20"]
ALL_SEASONS = WARMUP_SEASONS + SEASONS

REQUEST_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2  # waits 2s, 4s, 8s between attempts
PAUSE_BETWEEN_SEASONS_SECONDS = 1  # be polite to the NBA's servers

# Errors worth retrying: slow or dropped connections, server hiccups,
# and garbled responses that fail to parse as JSON.
RETRYABLE_ERRORS = (requests.exceptions.RequestException, ValueError, KeyError)


def fetch_season_game_log(season: str) -> pd.DataFrame:
    """Return one row per team per game for a regular season, e.g. season="2024-25"."""
    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",  # excludes preseason and playoffs
        player_or_team_abbreviation="T",  # "T" = team box scores, "P" = player box scores
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    # The endpoint returns a list of result tables; the game log is the first one.
    return response.get_data_frames()[0]


def fetch_with_retries(season: str) -> pd.DataFrame:
    """Call fetch_season_game_log, retrying with exponential backoff on failure."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fetch_season_game_log(season)
        except RETRYABLE_ERRORS as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Failed to download {season} after {MAX_ATTEMPTS} attempts"
                ) from error
            wait_seconds = BACKOFF_BASE_SECONDS ** attempt
            print(f"  attempt {attempt} failed ({type(error).__name__}); retrying in {wait_seconds}s")
            time.sleep(wait_seconds)


def raw_file_path(season: str) -> Path:
    return RAW_DATA_DIR / f"team_game_logs_{season}.csv"


def download_seasons(seasons: list[str], force: bool = False) -> None:
    """Download each season to data/raw/, skipping seasons already saved."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        output_path = raw_file_path(season)
        if output_path.exists() and not force:
            print(f"{season}: already downloaded, skipping ({output_path.name})")
            continue

        print(f"{season}: downloading...")
        game_log = fetch_with_retries(season)

        if game_log.empty:
            # Never write an empty file: it would look like a valid download later.
            print(f"{season}: API returned no rows, nothing saved")
        else:
            game_log.to_csv(output_path, index=False)
            print(
                f"{season}: saved {len(game_log)} team-game rows, "
                f"{game_log['GAME_ID'].nunique()} games -> {output_path.name}"
            )

        time.sleep(PAUSE_BETWEEN_SEASONS_SECONDS)


def load_raw_season(season: str) -> pd.DataFrame:
    """Read a saved raw season. GAME_ID is kept as text to preserve leading zeros."""
    return pd.read_csv(raw_file_path(season), dtype={"GAME_ID": str})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw NBA team game logs.")
    parser.add_argument("--force", action="store_true", help="re-download seasons already on disk")
    args = parser.parse_args()

    download_seasons(ALL_SEASONS, force=args.force)
