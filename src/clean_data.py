"""Turn raw team game logs into one row per game (data/processed/nba_games.csv).

The raw data has one row per team per game. This module checks its quality,
pairs each game's home and away rows, and saves one row per game with the
target column home_team_win.

Usage:
    python -m src.clean_data
"""

import pandas as pd

from src.collect_data import ALL_SEASONS, PROJECT_ROOT, load_raw_season

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GAMES_PATH = PROCESSED_DATA_DIR / "nba_games.csv"

# 2019-20 restart: after the COVID shutdown, every game from 2020-07-30 on was
# played without fans in the Orlando "bubble", so none had real home court.
BUBBLE_SEASON = "2019-20"
BUBBLE_START_DATE = "2020-07-30"

# Neutral-site games before 2024-25. In these seasons the API still lists a
# nominal home team, so they cannot be detected from MATCHUP. Each ID was
# checked against the raw data (teams and date match). From 2024-25 onward
# the API lists both teams as "@", which we detect automatically.
KNOWN_NEUTRAL_SITE_GAME_IDS = {
    "0022200439",  # 2022-12-17 MIA vs SAS, Mexico City
    "0022200678",  # 2023-01-19 CHI vs DET, Paris
    "0022300172",  # 2023-11-09 ATL vs ORL, Mexico City
    "0022301229",  # 2023-12-07 IND vs MIL, NBA Cup semifinal, Las Vegas
    "0022301230",  # 2023-12-07 NOP vs LAL, NBA Cup semifinal, Las Vegas
    "0022300527",  # 2024-01-11 BKN vs CLE, Paris
}

# Raw column -> clean column name, for the stats we keep for each side.
TEAM_STAT_COLUMNS = {
    "PTS": "score",
    "FG_PCT": "fg_pct",
    "FG3_PCT": "fg3_pct",
    "FT_PCT": "ft_pct",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    # Needed to estimate possessions for offensive / defensive ratings.
    "FGA": "fga",
    "FTA": "fta",
    "OREB": "oreb",
}


def load_all_raw_seasons(seasons: list[str] = ALL_SEASONS) -> pd.DataFrame:
    """Stack every raw season into one DataFrame, adding a readable SEASON column."""
    frames = [load_raw_season(season).assign(SEASON=season) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def add_home_away_flags(team_games: pd.DataFrame) -> pd.DataFrame:
    """Parse MATCHUP: "BOS vs. NYK" means BOS is home, "NYK @ BOS" means NYK is away."""
    return team_games.assign(
        IS_HOME=team_games["MATCHUP"].str.contains(" vs. ", regex=False),
        IS_AWAY=team_games["MATCHUP"].str.contains(" @ ", regex=False),
    )


def raw_data_quality_report(team_games: pd.DataFrame) -> dict:
    """Count problems in the raw team-game data. Every value should be 0 except where noted."""
    team_games = add_home_away_flags(team_games)
    per_game = team_games.groupby("GAME_ID")

    home_teams_per_game = per_game["IS_HOME"].sum()
    wins_per_game = per_game["WL"].apply(lambda results: (results == "W").sum())

    return {
        "team_game_rows": len(team_games),
        "games": team_games["GAME_ID"].nunique(),
        "exact_duplicate_rows": int(team_games.duplicated().sum()),
        "duplicate_game_team_rows": int(team_games.duplicated(["GAME_ID", "TEAM_ID"]).sum()),
        "games_without_two_rows": int((per_game.size() != 2).sum()),
        "games_without_two_distinct_teams": int((per_game["TEAM_ID"].nunique() != 2).sum()),
        "games_without_exactly_one_winner": int((wins_per_game != 1).sum()),
        "rows_with_unreadable_matchup": int((team_games["IS_HOME"] == team_games["IS_AWAY"]).sum()),
        "games_with_two_home_teams": int((home_teams_per_game == 2).sum()),
        # Not an error: the API lists both teams as "@" for some neutral-site games.
        "games_with_no_home_team": int((home_teams_per_game == 0).sum()),
        "missing_values_by_column": team_games.isna().sum()[lambda counts: counts > 0].to_dict(),
    }


def games_with_no_home_team(team_games: pd.DataFrame) -> pd.DataFrame:
    """Return the rows for games where neither team is listed as home."""
    team_games = add_home_away_flags(team_games)
    home_teams_per_game = team_games.groupby("GAME_ID")["IS_HOME"].sum()
    neutral_ids = home_teams_per_game[home_teams_per_game == 0].index
    columns = ["SEASON", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS"]
    return team_games.loc[team_games["GAME_ID"].isin(neutral_ids), columns].sort_values(
        ["GAME_DATE", "GAME_ID"]
    )


def assign_sides_for_api_neutral_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Give "both @" neutral-site games a home side so every game has one home row.

    Neither team truly had home court, so the choice is a labeling convention:
    the team whose abbreviation comes first alphabetically is labeled "home".
    These games are flagged with is_neutral_site = 1 so the label is never
    mistaken for real home-court advantage.
    """
    team_games = team_games.copy()
    home_teams_per_game = team_games.groupby("GAME_ID")["IS_HOME"].transform("sum")
    api_neutral = home_teams_per_game == 0

    first_abbreviation = team_games.groupby("GAME_ID")["TEAM_ABBREVIATION"].transform("min")
    team_games.loc[api_neutral, "IS_HOME"] = (
        team_games.loc[api_neutral, "TEAM_ABBREVIATION"] == first_abbreviation[api_neutral]
    )
    team_games["API_NEUTRAL_SITE"] = api_neutral
    return team_games


def side_table(team_games: pd.DataFrame, side: str) -> pd.DataFrame:
    """Select one side's rows ("home" or "away") and prefix its columns, e.g. PTS -> home_score."""
    is_side = team_games["IS_HOME"] if side == "home" else ~team_games["IS_HOME"]
    renames = {"TEAM_ABBREVIATION": f"{side}_team", "TEAM_ID": f"{side}_team_id"}
    renames.update({raw: f"{side}_{clean}" for raw, clean in TEAM_STAT_COLUMNS.items()})
    return team_games.loc[is_side, ["GAME_ID", *renames]].rename(columns=renames)


def build_games_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape team-game rows (two per game) into one row per game."""
    team_games = assign_sides_for_api_neutral_games(add_home_away_flags(raw))

    # Columns that describe the game itself (same on both rows); take them once.
    game_info = (
        team_games.groupby("GAME_ID")
        .agg(game_date=("GAME_DATE", "first"), season=("SEASON", "first"),
             api_neutral_site=("API_NEUTRAL_SITE", "any"))
        .reset_index()
    )

    games = (
        game_info
        .merge(side_table(team_games, "home"), on="GAME_ID", validate="one_to_one")
        .merge(side_table(team_games, "away"), on="GAME_ID", validate="one_to_one")
        .rename(columns={"GAME_ID": "game_id"})
    )

    games["game_date"] = pd.to_datetime(games["game_date"])
    is_bubble_game = (games["season"] == BUBBLE_SEASON) & (games["game_date"] >= BUBBLE_START_DATE)
    games["is_neutral_site"] = (
        games["api_neutral_site"]
        | games["game_id"].isin(KNOWN_NEUTRAL_SITE_GAME_IDS)
        | is_bubble_game
    ).astype(int)
    games["home_team_win"] = (games["home_score"] > games["away_score"]).astype(int)

    games = games.drop(columns="api_neutral_site")
    leading_columns = ["game_id", "game_date", "season", "home_team", "away_team",
                       "home_score", "away_score", "home_team_win", "is_neutral_site"]
    other_columns = [column for column in games.columns if column not in leading_columns]
    games = games[leading_columns + other_columns]

    # Same date can hold many games and we have no tip-off times, so sort by
    # date then game_id purely to make the file order stable and reproducible.
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def check_games_table(games: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Stop with a clear error if the reshaped table broke any basic rule."""
    problems = []
    if len(games) != raw["GAME_ID"].nunique():
        problems.append(f"expected {raw['GAME_ID'].nunique()} games, got {len(games)}")
    if games["game_id"].duplicated().any():
        problems.append("duplicate game_id values")
    if (games["home_team"] == games["away_team"]).any():
        problems.append("a team is listed as playing itself")
    if (games["home_score"] == games["away_score"]).any():
        problems.append("tied scores (NBA games cannot end tied)")

    # Cross-check the target against the API's own W/L column.
    home_won_per_api = raw.loc[raw["WL"] == "W"].set_index("GAME_ID")["TEAM_ABBREVIATION"]
    winners = games["home_team"].where(games["home_team_win"] == 1, games["away_team"])
    if not (winners.values == home_won_per_api.loc[games["game_id"]].values).all():
        problems.append("home_team_win disagrees with the API's W/L column")

    if problems:
        raise ValueError("Games table failed checks: " + "; ".join(problems))


def save_games_table(games: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_PATH, index=False)


def load_games_table() -> pd.DataFrame:
    """Read nba_games.csv with the right types (game_id as text, game_date as a date)."""
    return pd.read_csv(GAMES_PATH, dtype={"game_id": str}, parse_dates=["game_date"])


if __name__ == "__main__":
    raw = load_all_raw_seasons()

    print("Raw data quality report")
    print("-" * 40)
    for check, value in raw_data_quality_report(raw).items():
        print(f"{check:36s} {value}")

    games = build_games_table(raw)
    check_games_table(games, raw)
    save_games_table(games)

    print(f"\nSaved {len(games)} games -> {GAMES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Neutral-site games flagged: {games['is_neutral_site'].sum()}")
    print(f"Home team win rate: {games['home_team_win'].mean():.3f}")
    print()
    print(games.head(3).to_string(index=False))
