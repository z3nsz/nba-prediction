"""Accuracy experiments against the locked 10-input baseline (walk-forward, development seasons only).

    Experiment 1  probability that injury-listed players actually play
    Experiment 2  player impact: value above an average player, expected stars missing
    Experiment 3  Elo settings (K, season carry-over, home advantage, margin of victory),
                  re-tuned INSIDE each fold on that fold's training seasons only

Every model: logistic regression, no intercept, C = 1.0, same folds as the baseline.
Also: accuracy at confidence thresholds (a separate question from all-game accuracy).

Usage:
    python -m src.accuracy_experiments          (about 2 minutes)
"""

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.clean_data import load_games_table
from src.compare_models import expected_calibration_error
from src.evaluate import TARGET, score_predictions
from src.feature_engineering import compute_elo, elo_win_probability, load_features_table
from src.injury_impact import IMPACT_PATH
from src.logistic_model import predict_home_win, train_logistic
from src.model_data import PHASE2_INPUTS
from src.tuning import RESULTS_DIR
from src.walk_forward import FOLDS, date_block_bootstrap, development_only, with_availability

C = 1.0
BASELINE_INPUTS = PHASE2_INPUTS + ["out_minutes_diff", "questionable_minutes_diff", "starters_out_diff"]
EXPERIMENTS = {
    "baseline (10 inputs)": BASELINE_INPUTS,
    "E1a: questionable minutes x P(miss)": PHASE2_INPUTS + ["out_minutes_diff", "exp_missing_q_minutes_diff", "starters_out_diff"],
    "E1b: all listed minutes x P(miss)": PHASE2_INPUTS + ["exp_missing_minutes_diff", "starters_out_diff"],
    "E2a: baseline + missing value": BASELINE_INPUTS + ["exp_missing_value_diff"],
    "E2b: baseline + missing value + stars": BASELINE_INPUTS + ["exp_missing_value_diff", "exp_stars_missing_diff"],
    "E1+E2: P(miss) minutes + value + stars": PHASE2_INPUTS + ["exp_missing_minutes_diff", "starters_out_diff",
                                                              "exp_missing_value_diff", "exp_stars_missing_diff"],
}
ELO_GRID = {"k": [14, 18, 22, 26], "carryover": [0.5, 0.65, 0.75, 0.85, 1.0],
            "home_advantage": [40, 70, 100], "margin_of_victory": [True, False]}
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]


def load_experiment_features() -> pd.DataFrame:
    features = development_only(with_availability(load_features_table()))
    impact = pd.read_csv(IMPACT_PATH, dtype={"game_id": str})
    return features.merge(impact, on="game_id", how="left", validate="one_to_one")


def walk_forward(features: pd.DataFrame, columns: list[str], elo_by_fold: dict | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for each evaluated season. elo_by_fold replaces elo_diff per fold."""
    folds = []
    for train_seasons, eval_season in FOLDS:
        data = features if elo_by_fold is None else features.assign(
            elo_diff=features["game_id"].map(elo_by_fold[eval_season]))
        train = data[data["season"].isin(train_seasons)]
        evaluate = data[data["season"] == eval_season]
        model = train_logistic(train, columns, c=C)
        folds.append(evaluate[["game_id", "game_date", "season", TARGET]].assign(
            prob=predict_home_win(model, evaluate, columns)))
    return pd.concat(folds, ignore_index=True)


def tuned_elo_by_fold(games: pd.DataFrame, features: pd.DataFrame) -> tuple[dict, dict]:
    """For each fold: choose Elo settings by log loss on that fold's TRAINING seasons only."""
    combos = [dict(zip(ELO_GRID, values)) for values in itertools.product(*ELO_GRID.values())]
    runs = {}
    for combo in combos:
        elo = compute_elo(games, **combo).set_index("game_id")
        home_bonus = combo["home_advantage"] * (1 - games.set_index("game_id")["is_neutral_site"])
        runs[tuple(combo.values())] = (elo["home_elo"] - elo["away_elo"],
                                       elo_win_probability(elo["home_elo"] - elo["away_elo"] + home_bonus.loc[elo.index]))
    outcomes = features.set_index("game_id")[TARGET]
    chosen, elo_by_fold = {}, {}
    for train_seasons, eval_season in FOLDS:
        train_ids = features.loc[features["season"].isin(train_seasons), "game_id"]
        scores = {key: log_loss(outcomes.loc[train_ids], prob.loc[train_ids]) for key, (_, prob) in runs.items()}
        best = min(scores, key=scores.get)
        chosen[eval_season] = dict(zip(ELO_GRID, best))
        elo_by_fold[eval_season] = runs[best][0]
    return chosen, elo_by_fold


def describe(name: str, predictions: pd.DataFrame, reference: pd.DataFrame) -> dict:
    row = {"experiment": name, **score_predictions(predictions[TARGET], predictions["prob"]),
           "calibration_error": expected_calibration_error(predictions[TARGET], predictions["prob"])}
    folds_better = 0
    for season, fold in predictions.groupby("season"):
        ref = reference[reference["season"] == season]
        s, r = score_predictions(fold[TARGET], fold["prob"]), score_predictions(ref[TARGET], ref["prob"])
        row[f"acc {season}"], row[f"ll {season}"] = s["accuracy"], s["log_loss"]
        folds_better += s["log_loss"] < r["log_loss"] - 1e-12
    paired = predictions[["game_date", TARGET]].assign(new=predictions["prob"].to_numpy(), ref=reference["prob"].to_numpy())
    gap, low, high = date_block_bootstrap(paired, "new", "ref")
    row.update({"ll_vs_baseline": gap, "ci_low": low, "ci_high": high, "seasons_better": folds_better})
    return row


def confidence_table(predictions: pd.DataFrame) -> pd.DataFrame:
    confidence = np.maximum(predictions["prob"], 1 - predictions["prob"])
    correct = (predictions["prob"] > 0.5) == predictions[TARGET].astype(bool)
    return pd.DataFrame([{"min_confidence": t, "games_covered": (confidence >= t).mean(),
                          "accuracy": correct[confidence >= t].mean(), "games": int((confidence >= t).sum())}
                         for t in THRESHOLDS])


if __name__ == "__main__":
    features = load_experiment_features()
    predictions = {name: walk_forward(features, columns) for name, columns in EXPERIMENTS.items()}

    chosen, elo_by_fold = tuned_elo_by_fold(load_games_table(), features)
    predictions["E3: per-fold tuned Elo"] = walk_forward(features, BASELINE_INPUTS, elo_by_fold)
    print("Experiment 3, Elo settings chosen on each fold's training seasons:")
    for season, settings in chosen.items():
        print(f"  predicting {season}: {settings}")

    reference = predictions["baseline (10 inputs)"]
    table = pd.DataFrame([describe(name, p, reference) for name, p in predictions.items()])
    pd.set_option("display.width", 250)
    print("\nWalk-forward results (3,690 games; seasons 2022-23, 2023-24, 2024-25; test season not used):")
    print(table[["experiment", "accuracy", "log_loss", "brier", "auc", "calibration_error"]].round(4).to_string(index=False))
    print("\nPer season (accuracy / log loss):")
    print(table[["experiment"] + [c for c in table.columns if c.startswith(("acc ", "ll 2"))]].round(4).to_string(index=False))
    print("\nLog loss vs. baseline (negative = better), 95% date-block interval:")
    for _, r in table.iloc[1:].iterrows():
        verdict = "better" if r["ci_high"] < 0 else "worse" if r["ci_low"] > 0 else "no clear difference"
        print(f"  {r['experiment']:42s} {r['ll_vs_baseline']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
              f"better in {r['seasons_better']}/3 seasons  -> {verdict}")

    best_name = table.loc[table["log_loss"].idxmin(), "experiment"]
    for name in dict.fromkeys(["baseline (10 inputs)", best_name]):
        print(f"\nAccuracy at confidence thresholds: {name}")
        print(confidence_table(predictions[name]).round(3).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(5).to_csv(RESULTS_DIR / "accuracy_experiments.csv", index=False)
    (RESULTS_DIR / "accuracy_experiments_elo_choices.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print("\nSaved results/accuracy_experiments.csv")
