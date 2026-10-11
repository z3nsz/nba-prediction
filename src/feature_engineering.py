"""Build leakage-safe pre-game features (data/processed/nba_features.csv).

Every feature for a game uses only games that finished BEFORE it. The pattern:
    1. Reshape games into one row per team per game ("team-games").
    2. Within each team and season, sort by date, shift(1) so the current game
       is excluded, then take a rolling mean over the previous N games.
    3. Attach each team's features back onto the game as home_* / away_*.

Features:
    - recent form: win % (last 5 / 10), points scored / allowed and point
      differential (last 10)
    - rest: days since the team's previous game, back-to-back flag
    - home court: 1 for a true home game, 0 for a neutral-site game
    - Elo: each team's rating before tip-off and the implied home win probability
    - strength (src/strength_features.py): offensive / defensive / net rating per
      100 possessions, average opponent Elo, and margin versus Elo expectation
    - travel (src/travel_features.py): distance, 7-day distance, time-zone change,
      and position in the current road trip

Elo is computed over the warm-up seasons too, so ratings are realistic by the
start of 2020-21; only the modeling seasons are written to the features file.

Usage:
    python -m src.feature_engineering
"""

import pandas as pd

from src.clean_data import PROCESSED_DATA_DIR, load_games_table
from src.collect_data import SEASONS
from src.strength_features import STRENGTH_FEATURES, strength_features
from src.travel_features import TRAVEL_FEATURES, travel_features

FEATURES_PATH = PROCESSED_DATA_DIR / "nba_features.csv"

# (output name, team-game column, window size)
ROLLING_FEATURES = [
    ("win_pct_last5", "win", 5),
    ("win_pct_last10", "win", 10),
    ("avg_pts_scored_last10", "points_scored", 10),
    ("avg_pts_allowed_last10", "points_allowed", 10),
    ("avg_point_diff_last10", "point_diff", 10),
]

# Elo settings, taken from FiveThirtyEight's published NBA Elo method rather than
# tuned on our data, so choosing them does not peek at the seasons we test on.
ELO_START = 1500              # rating for a team we have not seen yet
ELO_K = 20                    # how far one game moves a rating
ELO_HOME_ADVANTAGE = 100      # rating points added to the home team (not at neutral sites)
ELO_SEASON_CARRYOVER = 0.75   # keep 75% of last season's distance from the mean...
ELO_SEASON_MEAN = 1505        # ...and regress the rest toward this value
# Known issue: +100 home advantage fits older NBA eras. In 2020-26 home teams win
# ~55%, so elo_home_win_prob overstates the home side by ~6-8 points. Do not
# use it as a probability as-is; calibrate it in Phase 2 on training seasons only.

# Columns copied from nba_games.csv into the features file. Box-score stats from
# the game itself (scores, FG%, rebounds...) are deliberately left out: they are
# only known after the game ends, so using them to predict it would be leakage.
GAME_COLUMNS = ["game_id", "game_date", "season", "home_team", "away_team",
                "is_neutral_site", "home_team_win"]


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per game into one row per team per game (two rows per game)."""
    home_side = pd.DataFrame({
        "game_id": games["game_id"],
        "game_date": games["game_date"],
        "season": games["season"],
        "team": games["home_team"],
        "side": "home",
        "points_scored": games["home_score"],
        "points_allowed": games["away_score"],
    })
    away_side = pd.DataFrame({
        "game_id": games["game_id"],
        "game_date": games["game_date"],
        "season": games["season"],
        "team": games["away_team"],
        "side": "away",
        "points_scored": games["away_score"],
        "points_allowed": games["home_score"],
    })
    team_games = pd.concat([home_side, away_side], ignore_index=True)
    team_games["win"] = (team_games["points_scored"] > team_games["points_allowed"]).astype(int)
    team_games["point_diff"] = team_games["points_scored"] - team_games["points_allowed"]

    # A team never plays twice on one date, so within a team, date order is game order.
    return team_games.sort_values(["team", "game_date"]).reset_index(drop=True)


def previous_games_mean(values: pd.Series, window: int) -> pd.Series:
    """Mean of up to `window` previous values. shift(1) drops the current game."""
    return values.shift(1).rolling(window, min_periods=1).mean()


def add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add pre-game rolling features, computed separately for each team and season.

    Grouping by season means history resets every October: a team's first game
    of a season has no prior games, so its features are NaN (left missing, never
    filled with later information). Rosters change between seasons, so last
    April's results are a weak guide to this October.
    """
    team_games = team_games.copy()
    by_team_season = team_games.groupby(["team", "season"])

    for feature_name, source_column, window in ROLLING_FEATURES:
        team_games[feature_name] = by_team_season[source_column].transform(
            previous_games_mean, window=window
        )

    # How many earlier games the averages are based on (0 = no history yet).
    # Lets a model learn that a 2-game average is less reliable than a 10-game one.
    team_games["games_played_before"] = by_team_season.cumcount()
    return team_games


def add_rest_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add days of rest and a back-to-back flag for each team-game.

    Game dates are published in the schedule long before tip-off, so using the
    current game's date here is not leakage.
    """
    team_games = team_games.copy()
    by_team_season = team_games.groupby(["team", "season"])

    # Days between this game and the team's previous game in the same season.
    # NaN for the season opener: there is no previous regular-season game.
    team_games["days_since_last_game"] = by_team_season["game_date"].diff().dt.days

    # 1 = played yesterday too. The opener is 0: no regular-season game came the day before.
    team_games["is_back_to_back"] = (team_games["days_since_last_game"] == 1).astype(int)
    return team_games


def elo_win_probability(rating_gap: float) -> float:
    """Chance the first team wins when it is rated `rating_gap` points higher."""
    return 1 / (1 + 10 ** (-rating_gap / 400))


def compute_elo(games: pd.DataFrame, k: float = ELO_K, home_advantage: float = ELO_HOME_ADVANTAGE,
                carryover: float = ELO_SEASON_CARRYOVER, margin_of_victory: bool = True) -> pd.DataFrame:
    """Walk through games in date order and record each team's Elo BEFORE each game.

    Ratings are only updated after a game's pre-game values are stored, so a
    game's own result never reaches its own features. Games on the same date
    never share a team, so the order of games within a date does not matter.
    `k` and `home_advantage` default to the published settings; src/elo_baseline.py
    tunes them on the training seasons.
    """
    games = games.sort_values(["game_date", "game_id"])
    ratings: dict[str, float] = {}
    current_season = None
    rows = []

    for game in games.itertuples():
        if game.season != current_season:
            # New season: pull every rating part of the way back toward average,
            # because rosters change over the summer.
            ratings = {team: ELO_SEASON_MEAN + carryover * (rating - ELO_SEASON_MEAN)
                       for team, rating in ratings.items()}
            current_season = game.season

        home_elo = ratings.get(game.home_team, ELO_START)
        away_elo = ratings.get(game.away_team, ELO_START)
        home_bonus = 0 if game.is_neutral_site else home_advantage
        home_win_prob = elo_win_probability(home_elo + home_bonus - away_elo)
        rows.append((game.game_id, home_elo, away_elo, home_win_prob))

        # Update after the game. Bigger wins move ratings more, but less so when
        # the favorite wins big (FiveThirtyEight's margin-of-victory multiplier).
        margin = abs(game.home_score - game.away_score)
        winner_gap = (home_elo + home_bonus - away_elo) * (1 if game.home_team_win else -1)
        margin_multiplier = (margin + 3) ** 0.8 / (7.5 + 0.006 * winner_gap) if margin_of_victory else 1.0
        change = k * margin_multiplier * (game.home_team_win - home_win_prob)
        ratings[game.home_team] = home_elo + change
        ratings[game.away_team] = away_elo - change

    return pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo", "elo_home_win_prob"])


def build_features_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game in the modeling seasons: game info, target, and pre-game features."""
    elo = compute_elo(games)
    team_games = add_rest_features(add_rolling_features(to_team_games(games)))
    team_games = team_games.merge(strength_features(games, elo, ELO_HOME_ADVANTAGE),
                                  on=["game_id", "team"], how="left", validate="one_to_one")
    team_games = team_games.merge(travel_features(games), on=["game_id", "team"], how="left", validate="one_to_one")
    feature_columns = [name for name, _, _ in ROLLING_FEATURES] + [
        "games_played_before", "days_since_last_game", "is_back_to_back", *STRENGTH_FEATURES, *TRAVEL_FEATURES,
    ]

    features = games[GAME_COLUMNS].copy()
    for side in ["home", "away"]:
        side_features = (
            team_games.loc[team_games["side"] == side, ["game_id", *feature_columns]]
            .rename(columns={column: f"{side}_{column}" for column in feature_columns})
        )
        features = features.merge(side_features, on="game_id", how="left", validate="one_to_one")

    # The home team has real home-court advantage unless the game was at a neutral site.
    features["home_court"] = 1 - features["is_neutral_site"]

    features = features.merge(elo, on="game_id", how="left", validate="one_to_one")
    features["elo_diff"] = features["home_elo"] - features["away_elo"]

    # Warm-up seasons only fed the Elo ratings; drop them from the output.
    features = features[features["season"].isin(SEASONS)]
    return features.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_features_table() -> pd.DataFrame:
    """Read nba_features.csv with the right types (game_id as text, game_date as a date)."""
    return pd.read_csv(FEATURES_PATH, dtype={"game_id": str}, parse_dates=["game_date"])


if __name__ == "__main__":
    games = load_games_table()
    features = build_features_table(games)
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Saved {len(features)} games x {features.shape[1]} columns -> data/processed/{FEATURES_PATH.name}")
    print("\nMissing values per feature (early-season games with no history):")
    missing = features.isna().sum()
    print(missing[missing > 0].to_string())

    print("\nHome win rate by rest situation (non-neutral games):")
    true_home = features[features["home_court"] == 1]
    situations = {
        "home on back-to-back, away rested": (true_home["home_is_back_to_back"] == 1) & (true_home["away_is_back_to_back"] == 0),
        "both rested or both back-to-back": true_home["home_is_back_to_back"] == true_home["away_is_back_to_back"],
        "away on back-to-back, home rested": (true_home["home_is_back_to_back"] == 0) & (true_home["away_is_back_to_back"] == 1),
    }
    for label, mask in situations.items():
        print(f"  {label:36s} {true_home.loc[mask, 'home_team_win'].mean():.3f}  ({mask.sum()} games)")

    elo_favorite_won = (features["elo_home_win_prob"] > 0.5).astype(int) == features["home_team_win"]
    print(f"\nElo favorite won {elo_favorite_won.mean():.3f} of games (always picking home: {features['home_team_win'].mean():.3f})")

    preview = ["game_date", "home_team", "away_team", "home_team_win",
               "home_elo", "away_elo", "elo_home_win_prob",
               "home_is_back_to_back", "away_is_back_to_back"]
    print("\nA mid-season sample:")
    print(features.loc[features["game_date"] == "2025-01-15", preview].head(4).round({"home_elo": 0, "away_elo": 0, "elo_home_win_prob": 3}).to_string(index=False))
