"""Hyperparameter tuning on the validation seasons.

For each model, every combination in a small grid is fit on TRAIN and scored
by log loss on VALIDATION; the best combination is kept. This is what the
validation split is for. Because validation now helps choose settings, its
scores become slightly optimistic, which is exactly why the test season stays
untouched until the final evaluation.

The chosen settings are saved to results/tuned_settings.json for later steps.

Usage:
    python -m src.tuning          (about 2 minutes)
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.collect_data import PROJECT_ROOT
from src.feature_engineering import load_features_table
from src.logistic_model import predict_home_win as logistic_predict, train_logistic
from src.model_data import FEATURE_SETS, x_and_y
from src.random_forest_model import predict_home_win as forest_predict, train_random_forest
from src.split import split_by_season
from src.xgboost_model import COLUMNS as TREE_COLUMNS, make_xgb

RESULTS_DIR = PROJECT_ROOT / "results"
TUNED_SETTINGS_PATH = RESULTS_DIR / "tuned_settings.json"

LOGISTIC_GRID = {
    "feature_set": list(FEATURE_SETS),
    "c": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
}
FOREST_GRID = {
    "min_samples_leaf": [10, 20, 35, 50, 80, 120],
    "max_features": [1, 2, 3, 5, 7],
    "max_depth": [None, 4, 8],
}
FOREST_TREES = 300  # fixed: more trees only smooths the average, it is not a choice to tune
XGB_GRID = {
    "max_depth": [1, 2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.1],
    "min_child_weight": [5, 20, 50],
}
XGB_TREE_COUNTS = [50, 100, 200, 300, 500, 800, 1200, 2000]


def grid_combinations(grid: dict) -> list[dict]:
    """Every combination of the grid's values, e.g. {"a": [1, 2], "b": [3]} -> [{a:1,b:3}, {a:2,b:3}]."""
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def tune_logistic(train, validation) -> pd.DataFrame:
    rows = []
    for combo in grid_combinations(LOGISTIC_GRID):
        columns = FEATURE_SETS[combo["feature_set"]]
        model = train_logistic(train, columns, c=combo["c"])
        probs = logistic_predict(model, validation, columns)
        rows.append({**combo, "validation_log_loss": log_loss(validation["home_team_win"], probs)})
    return pd.DataFrame(rows).sort_values("validation_log_loss").reset_index(drop=True)


def tune_random_forest(train, validation) -> pd.DataFrame:
    rows = []
    for combo in grid_combinations(FOREST_GRID):
        model = train_random_forest(train, {**combo, "n_estimators": FOREST_TREES})
        probs = forest_predict(model, validation)
        rows.append({**combo, "validation_log_loss": log_loss(validation["home_team_win"], probs)})
    return pd.DataFrame(rows).sort_values("validation_log_loss").reset_index(drop=True)


def tune_xgboost(train, validation) -> pd.DataFrame:
    """Fit each combination once with the most trees, then score every smaller tree count from it."""
    x_train, y_train = x_and_y(train, TREE_COLUMNS)
    x_val, y_val = x_and_y(validation, TREE_COLUMNS)
    rows = []
    for combo in grid_combinations(XGB_GRID):
        model = make_xgb({**combo, "subsample": 0.8, "colsample_bytree": 0.8},
                         n_estimators=max(XGB_TREE_COUNTS)).fit(x_train, y_train)
        for n_trees in XGB_TREE_COUNTS:
            # iteration_range=(0, n) uses only the first n trees: the same as a model with n trees.
            probs = model.predict_proba(x_val, iteration_range=(0, n_trees))[:, 1]
            rows.append({**combo, "n_estimators": n_trees, "validation_log_loss": log_loss(y_val, probs)})
    return pd.DataFrame(rows).sort_values("validation_log_loss").reset_index(drop=True)


def to_plain_value(value):
    """Undo pandas' type changes: NaN -> None, 7.0 -> 7 (scikit-learn reads max_features=7.0 as 700%)."""
    if hasattr(value, "item"):
        value = value.item()  # numpy number -> Python number
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    return value


def best_settings(results: pd.DataFrame) -> dict:
    """Top row of a tuning table as plain Python values, without the score."""
    best = results.iloc[0].drop("validation_log_loss").to_dict()
    return {key: to_plain_value(value) for key, value in best.items()}


if __name__ == "__main__":
    splits = split_by_season(load_features_table())
    train, validation = splits["train"], splits["validation"]

    tuners = {"logistic": tune_logistic, "random_forest": tune_random_forest, "xgboost": tune_xgboost}
    chosen, summary = {}, []
    for name, tune in tuners.items():
        results = tune(train, validation)
        chosen[name] = best_settings(results)
        print(f"\n{name}: tried {len(results)} settings. Best five (validation log loss):")
        print(results.head().round(4).to_string(index=False))
        summary.append({"model": name, "settings_tried": len(results),
                        "best_validation_log_loss": results["validation_log_loss"].iloc[0],
                        "fifth_best": results["validation_log_loss"].iloc[4]})

    print("\nSummary (validation log loss; tuned Elo baseline = 0.6105):")
    print(pd.DataFrame(summary).set_index("model").round(4).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    TUNED_SETTINGS_PATH.write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"\nSaved chosen settings -> {TUNED_SETTINGS_PATH.relative_to(PROJECT_ROOT)}")
