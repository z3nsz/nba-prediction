"""Walk-forward evaluation and feature-group ablation (Step 3.3).

Walk-forward validation imitates real use: train on every season before a
target season, predict that season, then move forward one season and repeat.

    fold 1: train 2020-21, 2021-22               -> predict 2022-23
    fold 2: train 2020-21 ... 2022-23            -> predict 2023-24
    fold 3: train 2020-21 ... 2023-24            -> predict 2024-25

The 2025-26 test season is never trained on, predicted, or scored here.

Ablation: start from the Phase 2 inputs and add ONE feature group at a time,
so each group's contribution is measured on its own.

Uncertainty: improvements are checked with a block bootstrap that resamples
whole game DATES, not single games, because games on the same night (and the
same teams across a season) are not independent.

Usage:
    python -m src.walk_forward          (about 1 minute)
"""

import json

import numpy as np
import pandas as pd

from src.compare_models import expected_calibration_error
from src.evaluate import TARGET, score_predictions
from src.feature_engineering import load_features_table
from src.logistic_model import predict_home_win as logistic_predict, train_logistic
from src.model_data import FEATURE_GROUPS, PHASE2_INPUTS
from src.random_forest_model import predict_home_win as forest_predict, train_random_forest
from src.split import SPLIT_SEASONS
from src.tuning import FOREST_TREES, RESULTS_DIR, TUNED_SETTINGS_PATH
from src.xgboost_model import predict_home_win as xgb_predict, train_xgboost

DEVELOPMENT_SEASONS = SPLIT_SEASONS["train"] + SPLIT_SEASONS["validation"]
TEST_SEASONS = SPLIT_SEASONS["test"]
FIRST_EVALUATED = "2022-23"  # need at least two seasons to train on
FOLDS = [(DEVELOPMENT_SEASONS[:i], season)
         for i, season in enumerate(DEVELOPMENT_SEASONS) if season >= FIRST_EVALUATED]
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

FEATURE_SET_CANDIDATES = {
    "phase 2 (7 inputs)": PHASE2_INPUTS,
    "+ ratings": PHASE2_INPUTS + FEATURE_GROUPS["ratings"],
    "+ schedule strength": PHASE2_INPUTS + FEATURE_GROUPS["schedule strength"],
    "+ ratings + schedule strength": PHASE2_INPUTS + FEATURE_GROUPS["ratings"] + FEATURE_GROUPS["schedule strength"],
    "+ availability": PHASE2_INPUTS + FEATURE_GROUPS["availability"],
    "+ travel": PHASE2_INPUTS + FEATURE_GROUPS["travel"],
    "+ availability + travel": PHASE2_INPUTS + FEATURE_GROUPS["availability"] + FEATURE_GROUPS["travel"],
    "+ availability + game score": PHASE2_INPUTS + FEATURE_GROUPS["availability"] + FEATURE_GROUPS["player impact"],
    "+ availability + plus-minus": PHASE2_INPUTS + FEATURE_GROUPS["availability"] + FEATURE_GROUPS["plus-minus impact"],
}
BASELINE = ("logistic", "phase 2 (7 inputs)")


def with_availability(features: pd.DataFrame) -> pd.DataFrame:
    """Attach injury-report features (development seasons only) if they have been built."""
    from src.availability_features import AVAILABILITY_PATH
    if not AVAILABILITY_PATH.exists():
        return features
    availability = pd.read_csv(AVAILABILITY_PATH, dtype={"game_id": str})
    return features.merge(availability, on="game_id", how="left", validate="one_to_one")


def development_only(features: pd.DataFrame) -> pd.DataFrame:
    """Drop the test season before anything else happens."""
    kept = features[features["season"].isin(DEVELOPMENT_SEASONS)].copy()
    assert not kept["season"].isin(TEST_SEASONS).any()
    return kept


def model_trainers(tuned: dict) -> dict:
    """name -> (train(train_df, columns), predict(model, df, columns))."""
    xgb_settings = {key: value for key, value in tuned["xgboost"].items() if key != "n_estimators"}
    xgb_settings.update({"subsample": 0.8, "colsample_bytree": 0.8})
    return {
        "logistic": (lambda train, cols: train_logistic(train, cols, c=tuned["logistic"]["c"]),
                     logistic_predict),
        "random forest": (lambda train, cols: train_random_forest(
                              train, {**tuned["random_forest"], "n_estimators": FOREST_TREES}, columns=cols),
                          lambda model, df, cols: forest_predict(model, df, columns=cols)),
        "xgboost": (lambda train, cols: train_xgboost(
                        train, xgb_settings, n_trees=tuned["xgboost"]["n_estimators"], columns=cols),
                    lambda model, df, cols: xgb_predict(model, df, columns=cols)),
    }


def walk_forward_predictions(features: pd.DataFrame, runs: list[tuple[str, str]], tuned: dict) -> pd.DataFrame:
    """Out-of-sample P(home wins) for every evaluated game, one column per (model, feature set)."""
    features = development_only(features)
    needs_availability = any("availability" in set_name for _, set_name in runs)
    if needs_availability and "home_out_minutes" not in features:
        raise ValueError("'+ availability' needs injury features: pass with_availability(load_features_table())")
    trainers = model_trainers(tuned)
    folds = []
    for train_seasons, eval_season in FOLDS:
        train = features[features["season"].isin(train_seasons)]
        evaluate = features[features["season"] == eval_season]
        fold = evaluate[["game_id", "game_date", "season", TARGET]].copy()
        for model_name, set_name in runs:
            fit, predict = trainers[model_name]
            columns = FEATURE_SET_CANDIDATES[set_name]
            fold[f"{model_name} | {set_name}"] = predict(fit(train, columns), evaluate, columns)
        folds.append(fold)
    return pd.concat(folds, ignore_index=True)


def date_block_bootstrap(predictions: pd.DataFrame, column: str, reference: str,
                         n_draws: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """95% interval for mean log loss (column - reference), resampling whole game dates."""
    y = predictions[TARGET].to_numpy()
    def per_game_loss(p):
        p = np.clip(predictions[p].to_numpy(), 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    gap = pd.Series(per_game_loss(column) - per_game_loss(reference), index=predictions.index)
    by_date = gap.groupby(predictions["game_date"]).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_date), size=(n_draws, len(by_date)))
    sums, sizes = by_date["sum"].to_numpy()[picks].sum(axis=1), by_date["size"].to_numpy()[picks].sum(axis=1)
    draws = sums / sizes
    return gap.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = " | ".join(BASELINE)
    has_reference = reference in predictions  # comparisons need the baseline among the runs
    rows = []
    for column in [c for c in predictions.columns if " | " in c]:
        model_name, set_name = column.split(" | ")
        pooled = score_predictions(predictions[TARGET], predictions[column])
        per_fold = predictions.groupby("season").apply(
            lambda fold: score_predictions(fold[TARGET], fold[column])["log_loss"])
        reference_per_fold = (predictions.groupby("season").apply(
            lambda fold: score_predictions(fold[TARGET], fold[reference])["log_loss"]) if has_reference else per_fold)
        row = {"model": model_name, "features": set_name, **pooled,
               "calibration_error": expected_calibration_error(predictions[TARGET], predictions[column]),
               **{f"log_loss {season}": value for season, value in per_fold.items()},
               "folds_better_than_baseline": int((per_fold < reference_per_fold - 1e-12).sum())}
        if has_reference and column != reference:
            gap, low, high = date_block_bootstrap(predictions, column, reference)
            row.update({"vs_baseline": gap, "ci_low": low, "ci_high": high})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tuned = json.loads(TUNED_SETTINGS_PATH.read_text())
    features = with_availability(load_features_table())
    candidates = [name for name in FEATURE_SET_CANDIDATES
                  if "availability" not in name or "home_out_minutes" in features]
    runs = ([("logistic", set_name) for set_name in candidates]
            + [(model, set_name) for model in ["random forest", "xgboost"]
               for set_name in ["phase 2 (7 inputs)", "+ ratings + schedule strength"]
               + [name for name in ["+ availability", "+ availability + travel", "+ availability + game score"]
                  if name in candidates]])
    predictions = walk_forward_predictions(features, runs, tuned)
    print("Folds:", ", ".join(f"{train[0]}..{train[-1]} -> {season}" for train, season in FOLDS))
    print(f"Evaluated games: {len(predictions)} (seasons {sorted(predictions['season'].unique())}; test season not used)\n")

    table = summarize(predictions)
    shown = ["model", "features", "log_loss", "brier", "accuracy", "auc", "calibration_error",
             "log_loss 2022-23", "log_loss 2023-24", "log_loss 2024-25", "folds_better_than_baseline"]
    print("Pooled walk-forward scores (all three evaluation seasons together):")
    print(table[shown].round(4).to_string(index=False))

    print("\nLog loss change vs. logistic | phase 2 (negative = better), 95% interval resampling game dates:")
    for _, row in table.dropna(subset=["vs_baseline"]).iterrows():
        verdict = ("better" if row["ci_high"] < 0 else "worse" if row["ci_low"] > 0 else "no clear difference")
        print(f"  {row['model']:13s} {row['features']:30s} {row['vs_baseline']:+.4f}  "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {verdict}")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(5).to_csv(RESULTS_DIR / "walk_forward_ablation.csv", index=False)
    print("\nSaved results/walk_forward_ablation.csv")
