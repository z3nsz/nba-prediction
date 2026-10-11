"""Phase 5, Step 1: where is the model reliable? (exploratory research)

Every game is viewed from the side of the team the MODEL favors:
    confidence = the model's probability for that team (always >= 50%)
    won        = 1 if that team won

For each category of games this reports the number of games, average predicted probability, actual win
rate (= pick accuracy), calibration gap (actual - predicted), log loss, and uncertainty:
    - Wilson 95% interval for the win rate (a standard interval for a proportion)
    - date-block bootstrap 95% interval for the calibration gap (games on the same night are not independent)

Datasets:
    walk-forward 2022-25  out-of-sample predictions from models trained only on earlier seasons (primary)
    2025-26               the final model's one-time test predictions (consistency check; already examined)

All categories use only information available 60 minutes before tip-off.

Usage:
    python -m src.category_analysis
"""

import numpy as np
import pandas as pd

from src.accuracy_experiments import EXPERIMENTS, load_experiment_features, walk_forward
from src.availability_features import AVAILABILITY_PATH, TEST_AVAILABILITY_PATH
from src.evaluate import TARGET
from src.feature_engineering import load_features_table
from src.injury_impact import IMPACT_PATH, TEST_IMPACT_PATH
from src.tuning import RESULTS_DIR

FINAL_MODEL = "E1+E2: P(miss) minutes + value + stars"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "final_test_predictions.csv"
N_BOOTSTRAP, SEED = 2000, 42


def walk_forward_predictions() -> pd.DataFrame:
    features = load_experiment_features()
    predictions = walk_forward(features, EXPERIMENTS[FINAL_MODEL])
    return predictions.rename(columns={"prob": "p_home"}).merge(
        features.drop(columns=[TARGET, "game_date", "season"]), on="game_id")


def test_season_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(TEST_PREDICTIONS_PATH, dtype={"game_id": str}, parse_dates=["game_date"])
    predictions = predictions.rename(columns={"E1+E2 (final model)": "p_home"})[["game_id", "game_date", TARGET, "p_home"]]
    extra = pd.concat([pd.read_csv(path, dtype={"game_id": str}) for path in [AVAILABILITY_PATH, TEST_AVAILABILITY_PATH]])
    impact = pd.concat([pd.read_csv(path, dtype={"game_id": str}) for path in [IMPACT_PATH, TEST_IMPACT_PATH]])
    features = load_features_table().merge(extra, on="game_id").merge(impact, on="game_id")
    return predictions.merge(features.drop(columns=[TARGET, "game_date"]), on="game_id")


def favorite_view(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game from the side of the model's favorite, with pre-game descriptors of that side."""
    home_fav = games["p_home"] >= 0.5
    pick = lambda home_col, away_col: np.where(home_fav, games[home_col], games[away_col])
    other = lambda home_col, away_col: np.where(home_fav, games[away_col], games[home_col])
    view = pd.DataFrame({
        "game_id": games["game_id"], "game_date": games["game_date"], "season": games["season"],
        "confidence": np.where(home_fav, games["p_home"], 1 - games["p_home"]),
        "won": np.where(home_fav, games[TARGET], 1 - games[TARGET]).astype(int),
        "favorite_is_home": home_fav & (games["home_court"] == 1),
        "neutral_site": games["home_court"] == 0,
        "elo_gap": np.where(home_fav, games["elo_diff"], -games["elo_diff"]),
        "fav_point_diff_last10": pick("home_avg_point_diff_last10", "away_avg_point_diff_last10"),
        "dog_point_diff_last10": other("home_avg_point_diff_last10", "away_avg_point_diff_last10"),
        "fav_starters_out": pick("home_starters_out", "away_starters_out"),
        "dog_starters_out": other("home_starters_out", "away_starters_out"),
        "questionable_minutes_total": games["home_questionable_minutes"].fillna(0) + games["away_questionable_minutes"].fillna(0),
        "fav_back_to_back": pick("home_is_back_to_back", "away_is_back_to_back"),
        "dog_back_to_back": other("home_is_back_to_back", "away_is_back_to_back"),
        "fav_games_played": pick("home_games_played_before", "away_games_played_before"),
    })
    return view


def categories(view: pd.DataFrame) -> dict[str, pd.Series]:
    """Each entry maps a dimension name to a label per game. Boundaries were fixed before looking at results."""
    elo_agrees = view["elo_gap"] > 0
    form_agrees = view["fav_point_diff_last10"] > view["dog_point_diff_last10"]
    return {
        "model confidence": pd.cut(view["confidence"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], include_lowest=True,
                                   labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]),
        "favorite strength": pd.cut(view["confidence"], [0.5, 0.6, 0.75, 1.0], include_lowest=True,
                                    labels=["slight (50-60%)", "moderate (60-75%)", "strong (75%+)"]),
        "Elo gap for the model's pick": pd.cut(view["elo_gap"], [-np.inf, 0, 50, 100, 200, np.inf],
                                              labels=["Elo disagrees (<0)", "0-50", "50-100", "100-200", "200+"]),
        "where the pick plays": pd.Series(np.select([view["neutral_site"], view["favorite_is_home"]],
                                                    ["neutral site", "pick at home"], "pick on the road"), index=view.index),
        "injury uncertainty (questionable minutes, both teams)": pd.cut(
            view["questionable_minutes_total"], [-0.1, 0, 30, np.inf], labels=["none", "1-30 min", "30+ min"]),
        "starters out": pd.Series(np.select(
            [(view["fav_starters_out"] > 0) & (view["dog_starters_out"] > 0), view["fav_starters_out"] > 0,
             view["dog_starters_out"] > 0], ["both teams", "pick missing a starter", "opponent missing a starter"],
            "neither"), index=view.index),
        "recent form (last 10 point diff)": pd.Series(np.select(
            [elo_agrees & form_agrees, elo_agrees & ~form_agrees], ["Elo and form agree", "form disagrees"],
            "Elo disagrees"), index=view.index),
        "rest": pd.Series(np.select(
            [view["fav_back_to_back"] == 1, view["dog_back_to_back"] == 1], ["pick on back-to-back", "opponent on back-to-back"],
            "neither on back-to-back"), index=view.index),
        "season stage": pd.cut(view["fav_games_played"], [-1, 14, 41, 100],
                               labels=["first 15 games", "games 16-42", "games 43+"]),
    }


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a win rate; better than +/- 2 standard errors for small groups or rates near 0/1."""
    if n == 0:
        return np.nan, np.nan
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def gap_interval(group: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    """95% date-block bootstrap interval for (actual win rate - average predicted probability)."""
    per_date = group.assign(gap=group["won"] - group["confidence"]).groupby("game_date")["gap"].agg(["sum", "size"])
    picks = rng.integers(0, len(per_date), size=(N_BOOTSTRAP, len(per_date)))
    draws = per_date["sum"].to_numpy()[picks].sum(axis=1) / per_date["size"].to_numpy()[picks].sum(axis=1)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def category_table(view: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for dimension, labels in categories(view).items():
        for label, group in view.groupby(labels, observed=True):
            if group.empty:
                continue
            low, high = wilson_interval(int(group["won"].sum()), len(group))
            gap_low, gap_high = gap_interval(group, rng)
            p = group["confidence"].clip(1e-6, 1 - 1e-6)
            rows.append({
                "dimension": dimension, "category": str(label), "games": len(group),
                "avg_predicted": group["confidence"].mean(), "actual_win_rate": group["won"].mean(),
                "win_rate_ci_low": low, "win_rate_ci_high": high,
                "calibration_gap": group["won"].mean() - group["confidence"].mean(),
                "gap_ci_low": gap_low, "gap_ci_high": gap_high,
                "log_loss": float(-(group["won"] * np.log(p) + (1 - group["won"]) * np.log(1 - p)).mean()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for name, games in [("walk-forward 2022-25 (primary)", walk_forward_predictions()),
                        ("2025-26 (consistency check)", test_season_predictions())]:
        view = favorite_view(games)
        table = category_table(view)
        print(f"\n===== {name}: {len(view)} games, pick accuracy {view['won'].mean():.1%} =====")
        for dimension, rows in table.groupby("dimension", sort=False):
            print(f"\n{dimension}")
            print(rows.drop(columns="dimension").round(3).to_string(index=False))
        flagged = table[(table["gap_ci_low"] > 0) | (table["gap_ci_high"] < 0)]
        print(f"\ncategories whose calibration-gap interval excludes 0: {len(flagged)} of {len(table)} "
              f"(about {0.05 * len(table):.1f} expected by chance alone)")
        table.round(5).to_csv(RESULTS_DIR / f"category_accuracy_{'walk_forward' if 'walk' in name else '2025_26'}.csv", index=False)
