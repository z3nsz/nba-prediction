"""Elo as a standalone baseline model, with settings tuned on the training seasons only.

Elo has two settings that matter most:
    home_advantage  rating points added to the home team (published value: 100)
    k               how far one game moves a rating (published value: 20)

We try a grid of both, run Elo through every game in date order, and keep the
pair with the lowest log loss on TRAINING games. The chosen settings are then
frozen and scored on validation. Ratings still update after every game, as
they would in real use: only the settings are frozen.

Usage:
    python -m src.elo_baseline
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.clean_data import load_games_table
from src.evaluate import TARGET, score_predictions
from src.feature_engineering import ELO_HOME_ADVANTAGE, ELO_K, compute_elo, load_features_table
from src.split import split_by_season

HOME_ADVANTAGE_GRID = range(0, 125, 5)
K_GRID = range(10, 42, 2)


def elo_home_win_probs(all_games: pd.DataFrame, k: float, home_advantage: float) -> pd.Series:
    """P(home wins) for every game, indexed by game_id."""
    elo = compute_elo(all_games, k=k, home_advantage=home_advantage)
    return elo.set_index("game_id")["elo_home_win_prob"]


def tune_elo(all_games: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Log loss on training games for every (k, home_advantage) pair, best first."""
    outcomes = train.set_index("game_id")[TARGET]
    results = []
    for home_advantage in HOME_ADVANTAGE_GRID:
        for k in K_GRID:
            probs = elo_home_win_probs(all_games, k, home_advantage).loc[outcomes.index]
            results.append({"home_advantage": home_advantage, "k": k,
                            "train_log_loss": log_loss(outcomes, probs)})
    return pd.DataFrame(results).sort_values("train_log_loss").reset_index(drop=True)


def calibration_table(outcomes: pd.Series, probs: pd.Series, bins=(0, .2, .3, .4, .5, .6, .7, .8, 1)) -> pd.DataFrame:
    """Average predicted vs. actual home win rate within probability ranges."""
    frame = pd.DataFrame({"predicted": probs.values, "actual": outcomes.values})
    return frame.groupby(pd.cut(frame["predicted"], bins), observed=True).agg(
        predicted=("predicted", "mean"), actual=("actual", "mean"), games=("actual", "size"))


if __name__ == "__main__":
    all_games = load_games_table()  # includes 2018-20 warm-up seasons, so ratings start realistic
    splits = split_by_season(load_features_table())

    grid = tune_elo(all_games, splits["train"])
    best = grid.iloc[0]
    print(f"Tried {len(grid)} settings on training games. Best five:")
    print(grid.head().round(4).to_string(index=False))

    versions = {
        f"Elo, published (k={ELO_K}, home={ELO_HOME_ADVANTAGE})": (ELO_K, ELO_HOME_ADVANTAGE),
        f"Elo, tuned on train (k={best.k:.0f}, home={best.home_advantage:.0f})": (best.k, best.home_advantage),
    }
    rows = []
    for name, (k, home_advantage) in versions.items():
        probs = elo_home_win_probs(all_games, k, home_advantage)
        for split_name in ["train", "validation"]:
            split = splits[split_name]
            rows.append({"model": name, "split": split_name,
                         **score_predictions(split[TARGET], probs.loc[split["game_id"]].values)})
    print("\nScores (test season not used):")
    print(pd.DataFrame(rows).set_index(["model", "split"]).round(4).to_string())

    validation = splits["validation"]
    for name, (k, home_advantage) in versions.items():
        probs = elo_home_win_probs(all_games, k, home_advantage).loc[validation["game_id"]]
        table = calibration_table(validation[TARGET], probs)
        gap = np.average(table["actual"] - table["predicted"], weights=table["games"])
        print(f"\nValidation calibration, {name}  (average actual minus predicted: {gap:+.3f})")
        print(table.round(3).to_string())
