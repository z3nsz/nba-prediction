"""Player-availability features from official pre-game injury reports (Step 3.2c).

For each team in each game:

    out_minutes           sum of expected minutes of players listed Out or Doubtful
    questionable_minutes  sum of expected minutes of players listed Questionable
    starters_out          players listed Out/Doubtful who averaged 28+ minutes
    out_game_score        sum of Out/Doubtful players' average Game Score (impact, not just minutes)
    questionable_game_score  the same for Questionable players
    out_plus_minus        sum of Out/Doubtful players' average plus-minus per game

Game Score (John Hollinger's box-score summary, ~20 for a star, ~5 for a bench player):
    PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA - FTM) + 0.7 OREB + 0.3 DREB + STL + 0.7 AST + 0.7 BLK - 0.4 PF - TOV

"Expected minutes" = the player's average minutes per game this season, using
only games on dates BEFORE this game. A player who has not played yet this
season counts as 0: his absence is already reflected in the team's results.

Leakage rules:
    - each game uses the report chosen by src/injury_download.py: the latest one
      printed at least 60 minutes before that game's tip-off (re-checked here)
    - expected minutes use strictly earlier dates (merge_asof, no exact matches)
    - box scores are never used to decide who was unavailable

Only development seasons have reports. Games without a usable report get NaN
(unknown), never a guessed value.

Usage:
    python -m src.availability_features
"""

import pandas as pd

from src.clean_data import PROCESSED_DATA_DIR, load_games_table
from src.collect_data import RAW_DATA_DIR
from src.injury_download import MANIFEST_PATH, PREGAME_BUFFER
from src.injury_reports import INJURY_REPORT_DIR, normalize_player_name, parse_report
from src.walk_forward import DEVELOPMENT_SEASONS

AVAILABILITY_PATH = PROCESSED_DATA_DIR / "availability_features.csv"
UNAVAILABLE = ["Out", "Doubtful"]
STARTER_MINUTES = 28
TEAM_ABBREVIATIONS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
VALUE_COLUMNS = ["out_minutes", "questionable_minutes", "starters_out",
                 "out_game_score", "questionable_game_score", "out_plus_minus"]
TEAM_COLUMNS = [*VALUE_COLUMNS, "report_found"]


def load_player_logs(seasons=DEVELOPMENT_SEASONS) -> pd.DataFrame:
    logs = pd.concat([pd.read_csv(RAW_DATA_DIR / f"player_game_logs_{season}.csv", parse_dates=["GAME_DATE"])
                      .assign(season=season) for season in seasons], ignore_index=True)
    logs["name_key"] = logs["PLAYER_NAME"].map(normalize_player_name)
    return logs


def game_score(logs: pd.DataFrame) -> pd.Series:
    """Hollinger's Game Score for each player box score."""
    return (logs["PTS"] + 0.4 * logs["FGM"] - 0.7 * logs["FGA"] - 0.4 * (logs["FTA"] - logs["FTM"])
            + 0.7 * logs["OREB"] + 0.3 * logs["DREB"] + logs["STL"] + 0.7 * logs["AST"] + 0.7 * logs["BLK"]
            - 0.4 * logs["PF"] - logs["TOV"])


PLAYER_STATS = {"MIN": "avg_minutes", "GAME_SCORE": "avg_game_score", "PLUS_MINUS": "avg_plus_minus"}


def expected_minutes_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Each player's season-to-date averages (minutes, Game Score, plus-minus) AFTER each date he played."""
    logs = logs.assign(GAME_SCORE=game_score(logs))
    per_player_day = (logs.groupby(["season", "PLAYER_ID", "name_key", "GAME_DATE"], as_index=False)[list(PLAYER_STATS)]
                      .sum().sort_values("GAME_DATE"))
    grouped = per_player_day.groupby(["season", "PLAYER_ID"])
    games_so_far = grouped.cumcount() + 1
    for stat, average in PLAYER_STATS.items():
        per_player_day[average] = grouped[stat].cumsum() / games_so_far
    return per_player_day[["season", "PLAYER_ID", "name_key", "GAME_DATE", *PLAYER_STATS.values()]]


PARSED_CACHE_PATH = PROCESSED_DATA_DIR / "injury_report_rows.csv"


def parsed_reports(files: list[str]) -> pd.DataFrame:
    """Parsed rows for the given report files. Each PDF is parsed once and cached (parsing is slow)."""
    wanted = sorted(set(files))
    if PARSED_CACHE_PATH.exists():
        cache = pd.read_csv(PARSED_CACHE_PATH, parse_dates=["report_time", "tip_off"])
    else:
        cache = pd.DataFrame(columns=["source_file"])
    missing = [name for name in wanted if name not in set(cache["source_file"])]
    if missing:
        new_rows = pd.concat([parse_report(INJURY_REPORT_DIR / name) for name in missing], ignore_index=True)
        cache = pd.concat([cache, new_rows], ignore_index=True) if len(cache) else new_rows
        cache.to_csv(PARSED_CACHE_PATH, index=False)
    return cache[cache["source_file"].isin(wanted)].reset_index(drop=True)


def attach_expected_minutes(listed: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """For each listed player, his average minutes from games strictly before the game date."""
    listed = listed.sort_values("game_date")
    minutes = minutes.sort_values("GAME_DATE")
    return pd.merge_asof(listed, minutes[["season", "name_key", "GAME_DATE", *PLAYER_STATS.values()]],
                         left_on="game_date", right_on="GAME_DATE", by=["season", "name_key"],
                         allow_exact_matches=False, direction="backward")


def listed_rows(games: pd.DataFrame, manifest: pd.DataFrame, seasons=DEVELOPMENT_SEASONS) -> pd.DataFrame:
    """Every row of each game's chosen pre-game report (players and NOT YET SUBMITTED markers)."""
    games = games[games["season"].isin(seasons)]
    manifest = manifest.dropna(subset=["report_file"])
    reports = parsed_reports(manifest["report_file"].tolist())
    reports["team_abbr"] = reports["team"].map(TEAM_ABBREVIATIONS)

    # Pair each game with the rows for ITS matchup in ITS chosen report.
    game_rows = (manifest[["game_id", "report_file"]]
                 .merge(games[["game_id", "game_date", "season", "home_team", "away_team"]], on="game_id")
                 .assign(matchup=lambda df: df["away_team"] + "@" + df["home_team"]))
    listed = game_rows.merge(reports, left_on=["report_file", "matchup"], right_on=["source_file", "matchup"],
                             how="inner", suffixes=("", "_report"))
    listed = listed[listed["tip_off"].isna() | (listed["tip_off"].dt.normalize() == listed["game_date"])]
    # Leakage guard: the report's printed time must be at least the buffer before tip-off.
    # A few games' tip-off times changed during the day (e.g. 8:30 PM in the morning report,
    # 7:30 PM later). Use the EARLIER time, and treat any game that fails as unknown.
    earliest_tip = listed.groupby("game_id")["tip_off"].transform("min").fillna(
        listed["game_id"].map(manifest.set_index("game_id")["tip_off"]))
    too_late_games = set(listed.loc[earliest_tip.notna() & (listed["report_time"] > earliest_tip - PREGAME_BUFFER), "game_id"])
    planned_tip = listed["game_id"].map(manifest.set_index("game_id")["tip_off"])
    too_late_games |= set(listed.loc[planned_tip.notna() & (listed["report_time"] > planned_tip - PREGAME_BUFFER), "game_id"])
    if too_late_games:
        print(f"  {len(too_late_games)} games: report too close to (possibly rescheduled) tip-off -> availability unknown")
    listed = listed[~listed["game_id"].isin(too_late_games)]
    assert not (listed["tip_off"].notna() & (listed["report_time"] > listed["tip_off"] - PREGAME_BUFFER)).any()
    return listed


def listed_players(listed: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Players in the chosen reports, with season-to-date averages from strictly earlier dates."""
    players = listed[listed["player"].notna()].copy()
    players["name_key"] = players["player"].map(normalize_player_name)
    players = attach_expected_minutes(players, expected_minutes_table(logs))
    players[list(PLAYER_STATS.values())] = players[list(PLAYER_STATS.values())].fillna(0)
    players["is_out"] = players["status"].isin(UNAVAILABLE)
    return players


def availability_features(games: pd.DataFrame, manifest: pd.DataFrame, logs: pd.DataFrame,
                          seasons=DEVELOPMENT_SEASONS) -> pd.DataFrame:
    """One row per game: home_/away_ availability features (NaN where no usable report)."""
    games = games[games["season"].isin(seasons)]
    listed = listed_rows(games, manifest, seasons)
    players = listed_players(listed, logs)
    team_totals = players.groupby(["game_id", "team_abbr"]).apply(lambda team: pd.Series({
        "out_minutes": team.loc[team["is_out"], "avg_minutes"].sum(),
        "questionable_minutes": team.loc[team["status"] == "Questionable", "avg_minutes"].sum(),
        "starters_out": int((team["is_out"] & (team["avg_minutes"] >= STARTER_MINUTES)).sum()),
        "out_game_score": team.loc[team["is_out"], "avg_game_score"].sum(),
        "questionable_game_score": team.loc[team["status"] == "Questionable", "avg_game_score"].sum(),
        "out_plus_minus": team.loc[team["is_out"], "avg_plus_minus"].sum(),
    }), include_groups=False).reset_index()

    # A team "has a report" if its matchup appeared and it was not marked NOT YET SUBMITTED.
    submitted = (listed[listed["status"] != "Not yet submitted"].groupby(["game_id", "team_abbr"]).size()
                 .rename("rows").reset_index())
    not_submitted = listed.loc[listed["status"] == "Not yet submitted", ["game_id", "team_abbr"]].drop_duplicates()
    matchups_found = set(listed["game_id"])

    result = games[["game_id"]].copy()
    for side in ["home", "away"]:
        keyed = games[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team_abbr"})
        side_values = keyed.merge(team_totals, on=["game_id", "team_abbr"], how="left")
        in_report = side_values["game_id"].isin(matchups_found)
        not_yet_submitted = (side_values[["game_id", "team_abbr"]]
                             .merge(not_submitted.assign(flag=True), on=["game_id", "team_abbr"], how="left")["flag"]
                             .notna().to_numpy())  # a real boolean array, so ~ means "not"
        known = in_report.to_numpy() & ~not_yet_submitted
        for column in VALUE_COLUMNS:
            # Listed in the report with no players = nobody injured = 0. Unknown report = NaN.
            side_values[column] = side_values[column].fillna(0).where(known)
        side_values["report_found"] = known.astype(int)
        result = result.merge(side_values.drop(columns="team_abbr").rename(
            columns={c: f"{side}_{c}" for c in TEAM_COLUMNS}), on="game_id")
    return result


TEST_AVAILABILITY_PATH = PROCESSED_DATA_DIR / "availability_features_test.csv"

if __name__ == "__main__":
    import argparse
    from src.injury_download import TEST_MANIFEST_PATH
    from src.walk_forward import TEST_SEASONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-season", action="store_true", help="build 2025-26 features (protocol committed)")
    args = parser.parse_args()
    seasons = TEST_SEASONS if args.test_season else DEVELOPMENT_SEASONS
    manifest = pd.read_csv(TEST_MANIFEST_PATH if args.test_season else MANIFEST_PATH,
                           dtype={"game_id": str}, parse_dates=["game_date", "tip_off"])
    table = availability_features(load_games_table(), manifest, load_player_logs(seasons), seasons)
    AVAILABILITY_PATH = TEST_AVAILABILITY_PATH if args.test_season else AVAILABILITY_PATH
    table.to_csv(AVAILABILITY_PATH, index=False)
    print(f"Saved {len(table)} games -> data/processed/{AVAILABILITY_PATH.name}")
    print("report found (home, away):", table["home_report_found"].mean().round(3), table["away_report_found"].mean().round(3))
    print(table.describe().T[["mean", "50%", "max"]].round(2).to_string())
