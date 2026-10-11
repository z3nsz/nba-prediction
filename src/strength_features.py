"""Pre-game team-strength features: efficiency ratings and strength of schedule.

Ratings (points per 100 possessions) remove the effect of pace. A team that
plays fast scores more points without being better; per-possession ratings
compare teams fairly.

    possessions (one team) ~ FGA + 0.44 * FTA - OREB + TOV
    game possessions       = average of both teams' estimates
    offensive rating       = 100 * points scored  / possessions
    defensive rating       = 100 * points allowed / possessions   (lower is better)
    net rating             = offensive - defensive

Rolling ratings are "ratios of sums": total points over the previous N games
divided by total possessions over the same games. That weights every
possession equally, instead of letting a slow 90-possession game count as much
as a fast 110-possession one.

Strength of schedule uses each opponent's Elo rating BEFORE that earlier game,
so it only contains information available at the time.

Every rolling value is shifted one game first, so the current game is never
included, and history resets each season (like the Phase 1 features).
"""

import pandas as pd

# FiveThirtyEight's published conversion: 28 Elo points ~ 1 point of expected margin.
ELO_POINTS_PER_POINT_OF_MARGIN = 28
WINDOWS = [5, 10]

STRENGTH_FEATURES = (
    [f"{stat}_last{window}" for window in WINDOWS for stat in ["off_rating", "def_rating", "net_rating"]]
    + [f"opp_elo_last{window}" for window in WINDOWS]
    + ["margin_vs_elo_last10"]
)


def team_box_scores(games: pd.DataFrame, elo: pd.DataFrame, home_advantage: float) -> pd.DataFrame:
    """One row per team per game with points, possessions, opponent Elo and Elo-expected margin."""
    games = games.merge(elo, on="game_id", validate="one_to_one")
    possessions = {
        side: games[f"{side}_fga"] + 0.44 * games[f"{side}_fta"] - games[f"{side}_oreb"] + games[f"{side}_tov"]
        for side in ["home", "away"]
    }
    game_possessions = (possessions["home"] + possessions["away"]) / 2
    home_bonus = home_advantage * (1 - games["is_neutral_site"])

    sides = []
    for side, other in [("home", "away"), ("away", "home")]:
        sign = 1 if side == "home" else -1
        sides.append(pd.DataFrame({
            "game_id": games["game_id"],
            "game_date": games["game_date"],
            "season": games["season"],
            "team": games[f"{side}_team"],
            "points_scored": games[f"{side}_score"],
            "points_allowed": games[f"{other}_score"],
            "possessions": game_possessions,
            "opp_elo": games[f"{other}_elo"],
            # Margin Elo expected for this team before the game (home bonus helps home, hurts away).
            "expected_margin": (games[f"{side}_elo"] - games[f"{other}_elo"] + sign * home_bonus)
                               / ELO_POINTS_PER_POINT_OF_MARGIN,
        }))
    box = pd.concat(sides, ignore_index=True)
    box["margin_vs_elo"] = (box["points_scored"] - box["points_allowed"]) - box["expected_margin"]
    return box.sort_values(["team", "game_date"]).reset_index(drop=True)


def previous_games_sum(values: pd.Series, window: int) -> pd.Series:
    """Sum of up to `window` previous values (current game excluded)."""
    return values.shift(1).rolling(window, min_periods=1).sum()


def previous_games_mean(values: pd.Series, window: int) -> pd.Series:
    return values.shift(1).rolling(window, min_periods=1).mean()


def strength_features(games: pd.DataFrame, elo: pd.DataFrame, home_advantage: float) -> pd.DataFrame:
    """Rolling pre-game ratings and schedule strength for every team-game.

    Returns game_id, team and the STRENGTH_FEATURES columns. A team's first game
    of a season has no history, so its values are NaN.
    """
    box = team_box_scores(games, elo, home_advantage)
    by_team_season = box.groupby(["team", "season"])

    for window in WINDOWS:
        points = by_team_season["points_scored"].transform(previous_games_sum, window=window)
        allowed = by_team_season["points_allowed"].transform(previous_games_sum, window=window)
        possessions = by_team_season["possessions"].transform(previous_games_sum, window=window)
        box[f"off_rating_last{window}"] = 100 * points / possessions
        box[f"def_rating_last{window}"] = 100 * allowed / possessions
        box[f"net_rating_last{window}"] = box[f"off_rating_last{window}"] - box[f"def_rating_last{window}"]
        box[f"opp_elo_last{window}"] = by_team_season["opp_elo"].transform(previous_games_mean, window=window)

    box["margin_vs_elo_last10"] = by_team_season["margin_vs_elo"].transform(previous_games_mean, window=10)
    return box[["game_id", "team", *STRENGTH_FEATURES]]
