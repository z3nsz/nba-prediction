"""Live data collection: injury reports, listed players, and season box scores.

    latest_report(cutoff)     the newest official injury report published at or before `cutoff`
                              (Eastern time). Final predictions use cutoff = tip-off - 60 minutes,
                              exactly like training; a morning preview uses "now".
    listed_for_games(...)     each game's listed players and which teams' reports are known
    update_box_scores(...)    append finished games' ESPN box scores to live_state/player_games_<season>.csv
    update_listings(...)      record each game's listed players in live_state/listings_<season>.csv,
                              later used to learn how often listed players actually play

Reports are published every 15 minutes (file names like Injury-Report_2026-10-20_05_00PM.pdf). The
search walks back in 15-minute steps; the report's PRINTED time is checked against the cutoff.
"""

import pandas as pd
import requests

from src.availability_features import TEAM_ABBREVIATIONS
from src.collect_data import RAW_DATA_DIR
from src.injury_reports import parse_report
from src.live.espn import box_score
from src.live.injury_features import BOX_COLUMNS, live_key
from src.live.team_features import STATE_DIR

LOG_COLUMNS = [*BOX_COLUMNS, "PLUS_MINUS"]   # plus-minus is averaged by the training code (not a model input)

REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
LIVE_REPORT_DIR = RAW_DATA_DIR / "injury_reports_live"
EARLIEST_REPORT_HOUR = 9          # no reports before 9:00 AM ET
SEARCH_BACK_HOURS = 30            # far enough to reach the previous evening
EASTERN = "America/New_York"


def report_stamp(when: pd.Timestamp) -> str:
    """Eastern time -> file stamp, e.g. 2026-10-20 17:00 -> '2026-10-20_05_00PM'."""
    return f"{when:%Y-%m-%d}_{when:%I_%M%p}"


def report_exists(stamp: str, session=requests) -> bool:
    try:
        return session.head(REPORT_URL.format(stamp=stamp), timeout=15).status_code == 200
    except requests.RequestException:
        return False


def latest_report(cutoff: pd.Timestamp, session=requests) -> str | None:
    """Stamp of the newest report labeled at or before `cutoff` (naive Eastern time), or None."""
    slot = cutoff.floor("15min")
    for _ in range(SEARCH_BACK_HOURS * 4):
        if slot.hour >= EARLIEST_REPORT_HOUR and report_exists(report_stamp(slot), session):
            return report_stamp(slot)
        slot -= pd.Timedelta(minutes=15)
    return None


def download_report(stamp: str, session=requests):
    path = LIVE_REPORT_DIR / f"Injury-Report_{stamp}.pdf"
    if not path.exists():
        response = session.get(REPORT_URL.format(stamp=stamp), timeout=30)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"{stamp}: not a PDF")
        LIVE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path


def listed_for_games(games: pd.DataFrame, report: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Listed players for `games` from one parsed report, and per-team "known" flags.

    games: game_id, game_date, home_team, away_team. A team is known if its game appears in the report and the
    team was not marked NOT YET SUBMITTED. The report must be printed at or before the cutoff (leakage guard).
    """
    if len(report) and report["report_time"].iloc[0] > cutoff:
        raise ValueError(f"report printed {report['report_time'].iloc[0]} is after the cutoff {cutoff}")
    report = report.assign(team_abbr=report["team"].map(TEAM_ABBREVIATIONS))
    report = report[report["tip_off"].dt.normalize().isin(games["game_date"].unique())]
    matchups = games.assign(matchup=games["away_team"] + "@" + games["home_team"])
    rows = matchups[["game_id", "matchup", "game_date"]].merge(report, on="matchup", suffixes=("", "_report"))
    rows = rows[rows["tip_off"].dt.normalize() == rows["game_date"]]
    listed = rows[rows["player"].notna()][["game_id", "team_abbr", "player", "status"]].reset_index(drop=True)

    known = []
    for side in ["home", "away"]:
        teams = matchups[["game_id", "matchup", f"{side}_team"]].rename(columns={f"{side}_team": "team_abbr"})
        found = teams["game_id"].isin(rows["game_id"])
        not_submitted = teams.merge(rows[rows["status"] == "Not yet submitted"][["game_id", "team_abbr"]].drop_duplicates()
                                    .assign(ns=True), on=["game_id", "team_abbr"], how="left")["ns"].notna().to_numpy()
        known.append(teams[["game_id", "team_abbr"]].assign(known=found.to_numpy() & ~not_submitted))
    return listed, pd.concat(known, ignore_index=True)


# ---------------------------------------------------------------- growing season files in live_state/

def box_scores_path(season: str):
    return STATE_DIR / f"player_games_{season}.csv"


def listings_path(season: str):
    return STATE_DIR / f"listings_{season}.csv"


def load_box_scores(season: str) -> pd.DataFrame:
    path = box_scores_path(season)
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "game_date", "team", "player", *LOG_COLUMNS])
    return pd.read_csv(path, dtype={"game_id": str}, parse_dates=["game_date"])


def update_box_scores(season: str, finished: pd.DataFrame) -> pd.DataFrame:
    """Add box scores for finished games not saved yet. finished: game_id ('espn-<id>'), game_date."""
    saved = load_box_scores(season)
    new = finished[~finished["game_id"].isin(saved["game_id"])]
    rows = [box_score(game.game_id.removeprefix("espn-"), cache=False).assign(game_id=game.game_id,
                                                                              game_date=game.game_date)
            for game in new.itertuples()]
    if rows:
        added = pd.concat(rows, ignore_index=True)[["game_id", "game_date", "team", "player", *LOG_COLUMNS]]
        saved = pd.concat([saved, added], ignore_index=True)
        saved.to_csv(box_scores_path(season), index=False)
    return saved


def season_logs(box_scores: pd.DataFrame, season: str) -> pd.DataFrame:
    """Saved box scores in the player-log format the injury features use."""
    key = box_scores["player"].map(live_key)
    return box_scores[["game_date", *LOG_COLUMNS]].rename(columns={"game_date": "GAME_DATE"}).assign(
        season=season, name_key=key, PLAYER_ID=key)


def load_listings(season: str) -> pd.DataFrame:
    path = listings_path(season)
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "game_date", "team_abbr", "player", "status", "report", "final"])
    return pd.read_csv(path, dtype={"game_id": str}, parse_dates=["game_date"])


def update_listings(season: str, listed: pd.DataFrame, games: pd.DataFrame, report: str, final: bool) -> pd.DataFrame:
    """Replace each game's saved listing with this one, unless a FINAL (60-minutes-before) listing exists."""
    saved = load_listings(season)
    finals = set(saved.loc[saved["final"].astype(bool), "game_id"])
    replace = [g for g in games["game_id"] if g not in finals]
    new = listed[listed["game_id"].isin(replace)].merge(games[["game_id", "game_date"]], on="game_id")
    saved = pd.concat([saved[~saved["game_id"].isin(replace)], new.assign(report=report, final=final)], ignore_index=True)
    saved.to_csv(listings_path(season), index=False)
    return saved


def season_listings(listings: pd.DataFrame, box_scores: pd.DataFrame) -> pd.DataFrame:
    """Past listings with whether the player then appeared in that game's box score (keys ignore suffixes)."""
    appeared = set(zip(box_scores["game_id"], box_scores["player"].map(live_key)))
    keys = listings["player"].map(live_key)
    played = [(g, k) in appeared for g, k in zip(listings["game_id"], keys)]
    return listings.assign(name_key=keys, played=played)[["game_date", "name_key", "status", "played", "game_id"]]


def parsed(stamp: str) -> pd.DataFrame:
    return parse_report(download_report(stamp))
