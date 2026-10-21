"""Live team features for the model: Elo, recent form, rest, back-to-backs, home court.

To avoid any difference from training, this runs the SAME code as the training pipeline
(src/feature_engineering.py: to_team_games, add_rolling_features, add_rest_features, compute_elo)
over all past games plus the current season's games from ESPN.

    live_state/history_games.csv   every game 2018-19 to 2025-26 (date, teams, score, neutral site);
                                   built once from our NBA data, committed so the cloud job can use it
    ESPN                           the current season: finished games (scores) and the day's schedule

Games without a final score yet are the ones being predicted. Their features use only games that
finished on earlier dates, exactly like training.

Usage:
    python -m src.live.team_features            (build live_state/history_games.csv)
"""

import numpy as np
import pandas as pd

from src.clean_data import load_games_table
from src.collect_data import PROJECT_ROOT
from src.feature_engineering import add_rest_features, add_rolling_features, compute_elo, to_team_games
from src.live.espn import REGULAR_SEASON, scoreboard
from src.travel_features import ARENAS

NBA_TEAMS = set(ARENAS)  # the 30 team codes
# ESPN labels these as regular season, but NBA statistics (our training data) do not count them.
NOT_REGULAR_SEASON_NOTES = ("NBA Cup Championship",)
UNPLAYED_STATUSES = ("Postponed", "Canceled", "Cancelled", "Suspended")

STATE_DIR = PROJECT_ROOT / "live_state"
HISTORY_PATH = STATE_DIR / "history_games.csv"
HISTORY_COLUMNS = ["game_id", "game_date", "season", "home_team", "away_team", "home_score", "away_score",
                   "home_team_win", "is_neutral_site"]
TEAM_INPUTS = ["elo_diff", "home_court", "point_diff_last10_diff", "win_pct_last10_diff", "win_pct_last5_diff",
               "rest_days_diff", "back_to_back_diff"]


def build_history() -> pd.DataFrame:
    history = load_games_table()[HISTORY_COLUMNS].copy()
    STATE_DIR.mkdir(exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)
    return history


def load_history() -> pd.DataFrame:
    return pd.read_csv(HISTORY_PATH, dtype={"game_id": str}, parse_dates=["game_date"])


def espn_season_games(season: str, first_date: pd.Timestamp, through_date: pd.Timestamp,
                      today: pd.Timestamp) -> pd.DataFrame:
    """Regular-season games from `first_date` to `through_date`, in the history format.

    Finished games keep their scores. On `through_date`, unfinished games have no score (they get predicted).
    Skipped, to match NBA regular-season statistics: All-Star games (non-NBA teams), the NBA Cup final,
    games not finished on earlier dates (postponed: ESPN lists them on the original date), and
    postponed or cancelled games on `through_date`.
    Dates before `today` are cached; today's scoreboard is always re-read.
    """
    rows = []
    for day in pd.date_range(first_date, through_date):
        board = scoreboard(day, cache=day < today)
        if board.empty:
            continue
        board = board[(board["season_type"] == REGULAR_SEASON)
                      & board["home_team"].isin(NBA_TEAMS) & board["away_team"].isin(NBA_TEAMS)
                      & ~board["note"].str.contains("|".join(NOT_REGULAR_SEASON_NOTES), regex=True)
                      & ~board["status"].isin(UNPLAYED_STATUSES)]
        if day < through_date:
            board = board[board["completed"]]
        for game in board.itertuples():
            done = game.completed
            rows.append({"game_id": f"espn-{game.espn_id}", "game_date": day, "season": season,
                         "home_team": game.home_team, "away_team": game.away_team,
                         "home_score": game.home_score if done else np.nan,
                         "away_score": game.away_score if done else np.nan,
                         "home_team_win": float(game.home_score > game.away_score) if done else np.nan,
                         "is_neutral_site": int(game.neutral_site), "tip_off_utc": game.tip_off_utc})
    return pd.DataFrame(rows)


def team_features(history: pd.DataFrame, season_games: pd.DataFrame) -> pd.DataFrame:
    """Model team inputs for every game in `season_games` that has no final score yet."""
    games = pd.concat([history, season_games.drop(columns="tip_off_utc", errors="ignore")], ignore_index=True)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    upcoming = games[games["home_score"].isna()]
    if upcoming.empty:
        return pd.DataFrame(columns=["game_id", *TEAM_INPUTS])

    team_games = add_rest_features(add_rolling_features(to_team_games(games)))
    side = lambda name: (team_games[team_games["side"] == name].set_index("game_id")
                         .loc[upcoming["game_id"]].reset_index())
    home, away = side("home"), side("away")
    elo = compute_elo(games).set_index("game_id").loc[upcoming["game_id"]].reset_index()

    # Same formulas as src/model_data.py (positive = favors the home team; missing history = 0).
    features = pd.DataFrame({
        "game_id": upcoming["game_id"].to_numpy(),
        "elo_diff": (elo["home_elo"] - elo["away_elo"]).to_numpy(),
        "home_court": (1 - upcoming["is_neutral_site"]).to_numpy(),
        "point_diff_last10_diff": (home["avg_point_diff_last10"] - away["avg_point_diff_last10"]).to_numpy(),
        "win_pct_last10_diff": (home["win_pct_last10"] - away["win_pct_last10"]).to_numpy(),
        "win_pct_last5_diff": (home["win_pct_last5"] - away["win_pct_last5"]).to_numpy(),
        "rest_days_diff": (home["days_since_last_game"] - away["days_since_last_game"]).to_numpy(),
        "back_to_back_diff": (away["is_back_to_back"] - home["is_back_to_back"]).to_numpy(),
    })
    return features.fillna(0)


if __name__ == "__main__":
    history = build_history()
    print(f"Saved {len(history)} games ({history['season'].iloc[0]} to {history['season'].iloc[-1]}) "
          f"-> {HISTORY_PATH.relative_to(PROJECT_ROOT)} ({HISTORY_PATH.stat().st_size / 1e3:.0f} KB)")
