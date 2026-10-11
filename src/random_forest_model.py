"""Random forest: many decision trees, each voting a probability.

A decision tree asks yes/no questions ("is elo_diff > 80?", "is the away team
on a back-to-back?") and ends in a leaf whose prediction is the home win rate
of the training games that landed there. A random forest trains hundreds of
trees, each on a random resample of games and a random subset of inputs at
each question, and averages their answers. Unlike logistic regression, trees
can find interactions (rest mattering more in close matchups) and curved
relationships without being told about them.

Trees do not need scaled inputs, so the model uses the same inputs as
logistic regression without a scaler.

Usage:
    python -m src.random_forest_model
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.evaluate import score_predictions
from src.feature_engineering import load_features_table
from src.model_data import FEATURE_SETS, x_and_y
from src.split import split_by_season

COLUMNS = FEATURE_SETS["elo + form + rest"]
RANDOM_SEED = 42  # fixes the randomness so results are reproducible

# scikit-learn defaults, as the untuned starting point.
DEFAULT_SETTINGS = {"n_estimators": 100, "min_samples_leaf": 1, "max_features": "sqrt"}

# Settings chosen by reasoning about the problem, not by searching on validation:
#   n_estimators=500    more trees = smoother averaged probabilities (only costs time)
#   min_samples_leaf=50 every leaf's win rate is based on at least 50 games. With
#                       1, each leaf memorizes one game and says 0% or 100%, which
#                       log loss punishes heavily. NBA games are noisy, so leaves
#                       need many games to estimate a probability.
#   max_features="sqrt" each question considers a random 2-3 of the 7 inputs, so
#                       trees differ from each other and their average is steadier
REASONED_SETTINGS = {"n_estimators": 500, "min_samples_leaf": 50, "max_features": "sqrt"}


def train_random_forest(train: pd.DataFrame, settings: dict, columns: list[str] = COLUMNS):
    x_train, y_train = x_and_y(train, columns)
    model = RandomForestClassifier(**settings, random_state=RANDOM_SEED, n_jobs=-1)
    return model.fit(x_train, y_train)


def predict_home_win(model, split: pd.DataFrame, columns: list[str] = COLUMNS):
    x, _ = x_and_y(split, columns)
    return model.predict_proba(x)[:, 1]


if __name__ == "__main__":
    splits = split_by_season(load_features_table())

    rows, models = [], {}
    for name, settings in {"default": DEFAULT_SETTINGS, "reasoned": REASONED_SETTINGS}.items():
        model = train_random_forest(splits["train"], settings)
        models[name] = model
        for split_name in ["train", "validation"]:
            split = splits[split_name]
            rows.append({"model": f"random forest: {name}", "split": split_name,
                         **score_predictions(split["home_team_win"], predict_home_win(model, split))})

    print("Random forest (fit on train; test season not used):\n")
    print(pd.DataFrame(rows).set_index(["model", "split"]).round(4).to_string())
    print("\nTo beat on validation: logistic (elo + form + rest) log loss 0.6062, Brier 0.2096, accuracy 0.6654")

    # Permutation importance: shuffle one input at a time on validation and measure
    # how much worse log loss gets. Bigger = the model relies on it more.
    x_val, y_val = x_and_y(splits["validation"], COLUMNS)
    importance = permutation_importance(models["reasoned"], x_val, y_val, scoring="neg_log_loss",
                                        n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
    print("\nPermutation importance, reasoned forest (increase in validation log loss when shuffled):")
    print(pd.Series(importance.importances_mean, index=COLUMNS).sort_values(ascending=False).round(4).to_string())
