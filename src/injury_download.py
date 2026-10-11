"""Download the official injury reports each development-season game needs (Step 3.2c).

For every game we want the LATEST report that was published at least
PREGAME_BUFFER before that game's tip-off. Tip-off times come from the reports
themselves, so for each game day:

    1. download the day's earliest report and read every game's tip-off time
    2. for each game, pick the latest report label whose publish time is at
       least PREGAME_BUFFER before tip-off, and download it
    3. if no same-day report is early enough (e.g. noon games when the first
       report is at 1:30 PM), use the previous day's last report, which also
       lists the next day's games

Only development seasons are downloaded; the test season is never requested.
Already-downloaded files are reused, so the script can be stopped and resumed.

Usage:
    python -m src.injury_download          (first run: ~20-40 minutes)
"""

import time

import pandas as pd
import requests

from src.clean_data import load_games_table
from src.collect_data import RAW_DATA_DIR
from src.injury_reports import INJURY_REPORT_DIR, parse_report
from src.walk_forward import DEVELOPMENT_SEASONS, TEST_SEASONS

BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{label}.pdf"
PREGAME_BUFFER = pd.Timedelta(minutes=60)
# Reports labeled "05PM" are stamped about 5:30 PM. Used only to choose which
# file to request; the timestamp printed inside the report is what gets checked.
LABEL_TO_PUBLISH_OFFSET = pd.Timedelta(minutes=30)
LABELS_2020_21 = ["01PM", "05PM", "08PM"]
LABELS_HOURLY = ["10AM", "11AM", "12PM", "01PM", "02PM", "03PM", "04PM", "05PM", "06PM", "07PM", "08PM", "09PM", "10PM"]
# From 2025-12-22 reports come every 15 minutes, named like "05_15PM", stamped at exactly that time.
QUARTER_HOUR_FORMAT_START = pd.Timestamp("2025-12-22")
LABELS_QUARTER_HOUR = [f"{(h - 1) % 12 + 1:02d}_{m:02d}{'AM' if h < 12 else 'PM'}"
                       for h in range(9, 23) for m in (0, 15, 30, 45)]
MANIFEST_PATH = RAW_DATA_DIR / "injury_report_manifest.csv"
TEST_MANIFEST_PATH = RAW_DATA_DIR / "injury_report_manifest_test.csv"
MAX_DOWNLOAD_BYTES = 100_000_000  # hard stop for one run; ask before going further
_downloaded_bytes = 0
PAUSE_SECONDS = 0.2
_failed_downloads: set[str] = set()


def labels_for(season: str, date: pd.Timestamp) -> list[str]:
    if season == "2020-21":
        return LABELS_2020_21
    return LABELS_QUARTER_HOUR if date >= QUARTER_HOUR_FORMAT_START else LABELS_HOURLY


def label_time(date: pd.Timestamp, label: str) -> pd.Timestamp:
    """Expected publish time: '05PM' (hourly era) -> 5:30 PM; '05_15PM' (15-minute era) -> 5:15 PM."""
    if "_" in label:
        return pd.to_datetime(f"{date:%Y-%m-%d} {label}", format="%Y-%m-%d %I_%M%p")
    return pd.to_datetime(f"{date:%Y-%m-%d} {label}", format="%Y-%m-%d %I%p") + LABEL_TO_PUBLISH_OFFSET


def report_path(date: pd.Timestamp, label: str):
    return INJURY_REPORT_DIR / f"Injury-Report_{date:%Y-%m-%d}_{label}.pdf"


def download(date: pd.Timestamp, label: str, session: requests.Session) -> bool:
    """Download one report if needed. Returns True if the file is now on disk."""
    global _downloaded_bytes
    path = report_path(date, label)
    if path.exists():
        return True
    if _downloaded_bytes > MAX_DOWNLOAD_BYTES:
        raise SystemExit(f"Stopped: this run downloaded {_downloaded_bytes / 1e6:.0f} MB, over the "
                         f"{MAX_DOWNLOAD_BYTES / 1e6:.0f} MB limit. Ask before continuing (re-running resumes).")
    if path.name in _failed_downloads:
        return False
    for attempt in range(3):
        try:
            response = session.get(BASE_URL.format(date=f"{date:%Y-%m-%d}", label=label), timeout=30)
            time.sleep(PAUSE_SECONDS)
            if response.status_code in (403, 404):  # the server answers 403 for files that do not exist
                _failed_downloads.add(path.name)
                return False
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            path.write_bytes(response.content)
            _downloaded_bytes += len(response.content)
            return True
        except (requests.RequestException, ValueError):
            time.sleep(2 ** (attempt + 1))
    _failed_downloads.add(path.name)
    return False


def first_available(date, labels, session, reverse=False):
    for label in (reversed(labels) if reverse else labels):
        if download(date, label, session):
            return label
    return None


def plan_and_download(games: pd.DataFrame) -> pd.DataFrame:
    """For every game, the report file it will use, and that report's printed time."""
    INJURY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    rows = []
    game_days = sorted(games["game_date"].unique())
    for day_number, day in enumerate(game_days, start=1):
        day = pd.Timestamp(day)
        season = games.loc[games["game_date"] == day, "season"].iloc[0]
        labels = labels_for(season, day)

        earliest = first_available(day, labels, session)
        tips = {}
        if earliest:
            listed = parse_report(report_path(day, earliest))
            listed = listed[listed["tip_off"].dt.normalize() == day]
            tips = listed.groupby("matchup")["tip_off"].first().to_dict()

        for game in games[games["game_date"] == day].itertuples():
            matchup = f"{game.away_team}@{game.home_team}"
            tip_off = tips.get(matchup)
            chosen_date, chosen_label = None, None
            if tip_off is not None:
                usable = [label for label in labels if label_time(day, label) <= tip_off - PREGAME_BUFFER]
                for label in reversed(usable):
                    if download(day, label, session):
                        chosen_date, chosen_label = day, label
                        break
            if chosen_label is None:  # fall back to the previous evening's last report
                previous = day - pd.Timedelta(days=1)
                label = first_available(previous, labels_for(season, previous), session, reverse=True)
                if label:
                    chosen_date, chosen_label = previous, label
            rows.append({"game_id": game.game_id, "game_date": day, "matchup": matchup, "tip_off": tip_off,
                         "report_file": report_path(chosen_date, chosen_label).name if chosen_label else None})
        if day_number % 50 == 0:
            print(f"  {day_number}/{len(game_days)} game days, {len(list(INJURY_REPORT_DIR.glob('*.pdf')))} files on disk")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download the injury reports each game needs.")
    parser.add_argument("--test-season", action="store_true",
                        help="download for the 2025-26 test season (only after the protocol is committed)")
    args = parser.parse_args()
    games = load_games_table()
    seasons = TEST_SEASONS if args.test_season else DEVELOPMENT_SEASONS
    games = games[games["season"].isin(seasons)]
    manifest = plan_and_download(games)
    manifest.to_csv(TEST_MANIFEST_PATH if args.test_season else MANIFEST_PATH, index=False)
    print(f"Downloaded this run: {_downloaded_bytes / 1e6:.1f} MB")
    files = list(INJURY_REPORT_DIR.glob("*.pdf"))
    print(f"\nGames: {len(manifest)} | with a report: {manifest['report_file'].notna().sum()} "
          f"| with a tip-off time: {manifest['tip_off'].notna().sum()}")
    print(f"Files on disk: {len(files)}, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")
