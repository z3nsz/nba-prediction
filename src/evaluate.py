"""Scoring rules for win-probability predictions, and the simplest possible baselines.

Every model in this project outputs P(home team wins) for each game. We score
those probabilities four ways:

    accuracy  share of games where the side given > 50% actually won (higher is better)
    log loss  average penalty -log(p of what happened); punishes confident misses (lower is better)
    brier     average squared error (p - outcome)^2 (lower is better)
    auc       how well predictions rank home wins above home losses (higher is better)

Baselines are scored on train and validation only. The test season stays
untouched until the final evaluation.

Usage:
    python -m src.evaluate
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.feature_engineering import load_features_table
from src.split import split_by_season

TARGET = "home_team_win"


def score_predictions(outcomes: pd.Series, home_win_probs: np.ndarray) -> dict[str, float]:
    """Score predicted home-win probabilities against actual results (1 = home won)."""
    home_win_probs = np.asarray(home_win_probs, dtype=float)
    # A pick is correct when the side given more than 50% won. An exact 50/50
    # prediction picks no one, so it earns half credit (what a random pick expects).
    picked_correctly = np.where(home_win_probs == 0.5, 0.5,
                                (home_win_probs > 0.5) == np.asarray(outcomes, dtype=bool))
    # AUC needs predictions that differ; a constant prediction cannot rank anything.
    auc = roc_auc_score(outcomes, home_win_probs) if np.ptp(home_win_probs) > 0 else 0.5
    return {
        "accuracy": float(picked_correctly.mean()),
        "log_loss": log_loss(outcomes, home_win_probs, labels=[0, 1]),
        "brier": brier_score_loss(outcomes, home_win_probs),
        "auc": auc,
    }


# ---------- Naive baselines ----------
# Each is "fit" on the training split only, then applied unchanged to later games.

def coin_flip_baseline(train: pd.DataFrame):
    """Know nothing: every game is 50/50."""
    return lambda games: np.full(len(games), 0.5)


def home_rate_baseline(train: pd.DataFrame):
    """Every game gets the training seasons' home win rate (always picks the home team)."""
    home_rate = train[TARGET].mean()
    return lambda games: np.full(len(games), home_rate)


def home_court_baseline(train: pd.DataFrame):
    """Training home win rate for true home games, 50% at neutral sites."""
    home_rate = train.loc[train["home_court"] == 1, TARGET].mean()
    return lambda games: np.where(games["home_court"] == 1, home_rate, 0.5)


BASELINES = {
    "coin flip (50%)": coin_flip_baseline,
    "home win rate": home_rate_baseline,
    "home win rate, neutral-aware": home_court_baseline,
}


def evaluate_baselines(splits: dict[str, pd.DataFrame], split_names=("train", "validation")) -> pd.DataFrame:
    """Fit every baseline on train and score it on each requested split."""
    rows = []
    for name, make_baseline in BASELINES.items():
        predict = make_baseline(splits["train"])
        for split_name in split_names:
            games = splits[split_name]
            rows.append({"model": name, "split": split_name,
                         **score_predictions(games[TARGET], predict(games))})
    return pd.DataFrame(rows).set_index(["model", "split"])


if __name__ == "__main__":
    splits = split_by_season(load_features_table())
    print("Naive baselines (fit on train; test season not used):\n")
    print(evaluate_baselines(splits).round(4).to_string())
