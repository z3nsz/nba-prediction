"""Compare every model on the validation seasons: scores, calibration, and uncertainty.

Every model uses its tuned settings (results/tuned_settings.json, and Elo tuned
on train), is fit on TRAIN, and predicts VALIDATION. Three questions:

    1. Which model scores best?               accuracy, log loss, Brier, AUC
    2. Are its probabilities honest?          calibration curve + expected calibration error
    3. Are the differences real or luck?      paired bootstrap of log loss differences

Outputs: results/validation_comparison.csv and results/validation_calibration.png

Usage:
    python -m src.compare_models          (about 30 seconds)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.clean_data import load_games_table
from src.elo_baseline import elo_home_win_probs, tune_elo
from src.evaluate import TARGET, home_rate_baseline, score_predictions
from src.feature_engineering import load_features_table
from src.logistic_model import predict_home_win as logistic_predict, train_logistic
from src.model_data import FEATURE_SETS
from src.random_forest_model import predict_home_win as forest_predict, train_random_forest
from src.split import split_by_season
from src.tuning import FOREST_TREES, RESULTS_DIR, TUNED_SETTINGS_PATH
from src.xgboost_model import predict_home_win as xgb_predict, train_xgboost

N_CALIBRATION_BINS = 10
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
REFERENCE_MODEL = "logistic regression"
COLORS = {"always home": "#52514e", "Elo (tuned)": "#eb6834", "logistic regression": "#2a78d6",
          "random forest": "#1baf7a", "XGBoost": "#eda100"}


def predictions_for(target_split: pd.DataFrame, splits: dict, all_games: pd.DataFrame,
                    tuned: dict, elo_settings: dict) -> pd.DataFrame:
    """P(home wins) from every model for each game in `target_split`, all fit on train."""
    train = splits["train"]
    logistic_columns = FEATURE_SETS[tuned["logistic"]["feature_set"]]
    xgb_settings = {key: value for key, value in tuned["xgboost"].items() if key != "n_estimators"}
    elo = elo_home_win_probs(all_games, **elo_settings)
    return pd.DataFrame({
        "always home": home_rate_baseline(train)(target_split),
        "Elo (tuned)": elo.loc[target_split["game_id"]].values,
        "logistic regression": logistic_predict(
            train_logistic(train, logistic_columns, c=tuned["logistic"]["c"]), target_split, logistic_columns),
        "random forest": forest_predict(
            train_random_forest(train, {**tuned["random_forest"], "n_estimators": FOREST_TREES}), target_split),
        "XGBoost": xgb_predict(
            train_xgboost(train, {**xgb_settings, "subsample": 0.8, "colsample_bytree": 0.8},
                          n_trees=tuned["xgboost"]["n_estimators"]), target_split),
    }, index=target_split.index)


def calibration_bins(outcomes: pd.Series, probs: pd.Series, n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    """Split games into equal-size groups by predicted probability; compare predicted vs. actual."""
    frame = pd.DataFrame({"predicted": np.asarray(probs), "actual": np.asarray(outcomes)})
    groups = pd.qcut(frame["predicted"].rank(method="first"), n_bins, labels=False)
    return frame.groupby(groups).agg(predicted=("predicted", "mean"), actual=("actual", "mean"),
                                     games=("actual", "size"))


def expected_calibration_error(outcomes: pd.Series, probs: pd.Series) -> float:
    """Average gap between predicted and actual win rate across the groups, weighted by games."""
    bins = calibration_bins(outcomes, probs)
    return float(np.average((bins["actual"] - bins["predicted"]).abs(), weights=bins["games"]))


def bootstrap_log_loss_gap(outcomes: pd.Series, model_probs, reference_probs,
                           n_draws: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """95% interval for (model log loss - reference log loss), resampling games with replacement.

    Both models are scored on the SAME resampled games each time ("paired"), so
    the interval reflects how much the gap could change with a different set of games.
    """
    outcomes = np.asarray(outcomes)
    clip = lambda p: np.clip(np.asarray(p), 1e-15, 1 - 1e-15)
    per_game_gap = (-(outcomes * np.log(clip(model_probs)) + (1 - outcomes) * np.log(1 - clip(model_probs)))
                    + (outcomes * np.log(clip(reference_probs)) + (1 - outcomes) * np.log(1 - clip(reference_probs))))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_game_gap), size=(n_draws, len(per_game_gap)))
    gaps = per_game_gap[draws].mean(axis=1)
    return per_game_gap.mean(), np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)


def plot_calibration(outcomes: pd.Series, predictions: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color="#52514e", linestyle="--", linewidth=1, label="perfect calibration")
    for model in ["Elo (tuned)", "logistic regression", "random forest", "XGBoost"]:
        bins = calibration_bins(outcomes, predictions[model])
        ax.plot(bins["predicted"], bins["actual"], marker="o", markersize=6, linewidth=2,
                color=COLORS[model], label=model)
    ax.set(xlim=(0.1, 0.9), ylim=(0.1, 0.9), xlabel="Predicted P(home wins), 10 equal-size groups",
           ylabel="Actual home win rate", title="Validation calibration (2023-24, 2024-25)")
    ax.set_aspect("equal")
    ax.grid(color="#e6e5e0")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    all_games = load_games_table()
    splits = split_by_season(load_features_table())
    tuned = json.loads(TUNED_SETTINGS_PATH.read_text())
    best_elo = tune_elo(all_games, splits["train"]).iloc[0]
    elo_settings = {"k": best_elo["k"], "home_advantage": best_elo["home_advantage"]}

    validation = splits["validation"]
    outcomes = validation[TARGET]
    predictions = predictions_for(validation, splits, all_games, tuned, elo_settings)

    rows = []
    for model in predictions:
        row = {"model": model, **score_predictions(outcomes, predictions[model]),
               "calibration_error": expected_calibration_error(outcomes, predictions[model])}
        if model != REFERENCE_MODEL:
            gap, low, high = bootstrap_log_loss_gap(outcomes, predictions[model], predictions[REFERENCE_MODEL])
            row.update({"log_loss_vs_logistic": gap, "gap_95_low": low, "gap_95_high": high})
        rows.append(row)
    comparison = pd.DataFrame(rows).set_index("model").sort_values("log_loss")

    print("Validation comparison (all models fit on train; test season not used)\n")
    print(comparison[["accuracy", "log_loss", "brier", "auc", "calibration_error"]].round(4).to_string())
    print(f"\nLog loss gap vs. {REFERENCE_MODEL} (positive = worse), with 95% bootstrap interval:")
    for model, row in comparison.drop(index=REFERENCE_MODEL).iterrows():
        verdict = "clearly worse" if row["gap_95_low"] > 0 else "not clearly different"
        print(f"  {model:15s} {row['log_loss_vs_logistic']:+.4f}  [{row['gap_95_low']:+.4f}, {row['gap_95_high']:+.4f}]  {verdict}")

    print(f"\nLogistic regression calibration on validation ({N_CALIBRATION_BINS} equal-size groups):")
    print(calibration_bins(outcomes, predictions[REFERENCE_MODEL]).round(3).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    comparison.round(5).to_csv(RESULTS_DIR / "validation_comparison.csv")
    plot_calibration(outcomes, predictions, RESULTS_DIR / "validation_calibration.png")
    print("\nSaved results/validation_comparison.csv and results/validation_calibration.png")
