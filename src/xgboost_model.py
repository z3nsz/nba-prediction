"""XGBoost: gradient-boosted trees.

A random forest grows many independent trees and averages them. Boosting grows
small trees ONE AFTER ANOTHER: each new tree is fit to the mistakes the trees
so far still make, and its answer is added in, scaled down by a learning rate.
Many small, careful corrections add up to a flexible model.

How many trees to use is chosen by EARLY STOPPING without touching validation:
    1. fit on 2020-21 + 2021-22, watch log loss on 2022-23 (the last training season)
    2. stop adding trees once 2022-23 log loss stops improving
    3. refit on all three training seasons with that number of trees
Validation stays untouched for comparing models.

Known limitation: step 1 sees only two seasons, and with less data the best
tree count is lower, so the refit model ends up slightly under-trained (its
predictions are too close to 50%). The tree count is tuned properly in Step 2.7.

Usage:
    python -m src.xgboost_model
"""

import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

from src.evaluate import score_predictions
from src.feature_engineering import load_features_table
from src.model_data import FEATURE_SETS, x_and_y
from src.split import SPLIT_SEASONS, split_by_season

COLUMNS = FEATURE_SETS["elo + form + rest"]
RANDOM_SEED = 42
EARLY_STOPPING_SEASON = SPLIT_SEASONS["train"][-1]  # 2022-23
MAX_TREES = 2000

# XGBoost's defaults, as the untuned starting point (100 trees, depth 6, learning rate 0.3).
DEFAULT_SETTINGS = {"n_estimators": 100}

# Settings chosen by reasoning, not by searching on validation:
#   max_depth=2          each tree asks at most 2 questions, so it can capture
#                        simple interactions (rest x team strength) but not memorize
#   learning_rate=0.03   each tree's correction is scaled down to 3%: many small
#                        steps generalize better than a few big ones
#   min_child_weight=20  a split must leave enough games on each side (the
#                        boosting version of min_samples_leaf)
#   subsample=0.8        each tree sees a random 80% of games...
#   colsample_bytree=0.8 ...and a random 80% of inputs, adding variety like a forest
#   reg_lambda=1.0       mild penalty on large leaf values (XGBoost's default)
REASONED_SETTINGS = {
    "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 20,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
}


def make_xgb(settings: dict, **extra) -> XGBClassifier:
    return XGBClassifier(**settings, **extra, objective="binary:logistic", eval_metric="logloss",
                         random_state=RANDOM_SEED, n_jobs=-1)


def choose_number_of_trees(train: pd.DataFrame, settings: dict, columns: list[str] = COLUMNS) -> int:
    """Early stopping inside the training seasons: fit on earlier seasons, watch the last one."""
    inner_fit = train[train["season"] != EARLY_STOPPING_SEASON]
    inner_check = train[train["season"] == EARLY_STOPPING_SEASON]
    x_fit, y_fit = x_and_y(inner_fit, columns)
    x_check, y_check = x_and_y(inner_check, columns)
    model = make_xgb(settings, n_estimators=MAX_TREES, early_stopping_rounds=100)
    model.fit(x_fit, y_fit, eval_set=[(x_check, y_check)], verbose=False)
    return model.best_iteration + 1


def train_xgboost(train: pd.DataFrame, settings: dict, n_trees: int | None = None,
                  columns: list[str] = COLUMNS) -> XGBClassifier:
    """Fit on all training seasons. If n_trees is given it overrides settings["n_estimators"]."""
    x_train, y_train = x_and_y(train, columns)
    extra = {"n_estimators": n_trees} if n_trees else {}
    return make_xgb(settings, **extra).fit(x_train, y_train)


def predict_home_win(model, split: pd.DataFrame, columns: list[str] = COLUMNS):
    x, _ = x_and_y(split, columns)
    return model.predict_proba(x)[:, 1]


if __name__ == "__main__":
    splits = split_by_season(load_features_table())

    n_trees = choose_number_of_trees(splits["train"], REASONED_SETTINGS)
    print(f"Early stopping on {EARLY_STOPPING_SEASON} (inside train) chose {n_trees} trees\n")

    models = {
        "default": train_xgboost(splits["train"], DEFAULT_SETTINGS),
        "reasoned": train_xgboost(splits["train"], REASONED_SETTINGS, n_trees=n_trees),
    }
    rows = []
    for name, model in models.items():
        for split_name in ["train", "validation"]:
            split = splits[split_name]
            rows.append({"model": f"xgboost: {name}", "split": split_name,
                         **score_predictions(split["home_team_win"], predict_home_win(model, split))})

    print("XGBoost (fit on train; test season not used):\n")
    print(pd.DataFrame(rows).set_index(["model", "split"]).round(4).to_string())
    print("\nTo beat on validation: logistic (elo + form + rest) log loss 0.6062, Brier 0.2096, accuracy 0.6654")

    x_val, y_val = x_and_y(splits["validation"], COLUMNS)
    importance = permutation_importance(models["reasoned"], x_val, y_val, scoring="neg_log_loss",
                                        n_repeats=10, random_state=RANDOM_SEED)
    print("\nPermutation importance, reasoned XGBoost (increase in validation log loss when shuffled):")
    print(pd.Series(importance.importances_mean, index=COLUMNS).sort_values(ascending=False).round(4).to_string())
