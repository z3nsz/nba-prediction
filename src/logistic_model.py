"""Logistic regression: our first trained model.

Logistic regression learns one weight per input and turns their weighted sum
into a probability with the sigmoid curve:

    P(home wins) = 1 / (1 + e^-(w1*x1 + w2*x2 + ...))

Design choices:
- No intercept. Every input except home_court is a home-minus-away difference,
  so swapping the two teams flips its sign. Without an intercept, two equal
  teams at a neutral site get exactly 50%, and home_court (1 or 0) plays the
  role of the intercept: its weight IS the learned home-court advantage,
  estimated from thousands of real home games. (With an intercept, the model
  over-learned from just 2 neutral-site training games.)
- Difference inputs are divided by their standard deviation so the weights are
  comparable. They are not shifted to mean 0 (with_mean=False), because that
  would move "equal teams" away from 0 and break the symmetry above.
- Scaling statistics come from TRAINING data only, inside a scikit-learn
  Pipeline, so nothing about later seasons leaks in.

Usage:
    python -m src.logistic_model
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import score_predictions
from src.feature_engineering import load_features_table
from src.model_data import FEATURE_SETS, x_and_y
from src.split import split_by_season

# C is the inverse of regularization strength: smaller C pulls weights toward 0
# to avoid overfitting. 1.0 is scikit-learn's default; it is tuned in Step 2.7.
REGULARIZATION_C = 1.0


def train_logistic(train: pd.DataFrame, columns: list[str], c: float = REGULARIZATION_C):
    """Fit scale-differences -> logistic regression (no intercept) on the training split."""
    x_train, y_train = x_and_y(train, columns)
    difference_columns = [column for column in columns if column != "home_court"]
    scale_differences = ColumnTransformer(
        [("scale", StandardScaler(with_mean=False), difference_columns)],
        remainder="passthrough",  # home_court stays 0/1
        verbose_feature_names_out=False,
    )
    model = make_pipeline(scale_differences, LogisticRegression(C=c, fit_intercept=False, max_iter=1000))
    return model.fit(x_train, y_train)


def predict_home_win(model, split: pd.DataFrame, columns: list[str]):
    """P(home wins) for each game in `split`."""
    x, _ = x_and_y(split, columns)
    return model.predict_proba(x)[:, 1]  # column 1 = probability of class 1 (home win)


def standardized_weights(model) -> pd.Series:
    """Learned weight per input: per one standard deviation for differences, per game for home_court."""
    names = model[0].get_feature_names_out()
    return pd.Series(model[-1].coef_[0], index=names).sort_values(key=abs, ascending=False)


if __name__ == "__main__":
    splits = split_by_season(load_features_table())

    rows, models = [], {}
    for set_name, columns in FEATURE_SETS.items():
        model = train_logistic(splits["train"], columns)
        models[set_name] = model
        for split_name in ["train", "validation"]:
            split = splits[split_name]
            rows.append({"model": f"logistic: {set_name}", "split": split_name,
                         **score_predictions(split["home_team_win"], predict_home_win(model, split, columns))})

    print("Logistic regression (fit on train; test season not used):\n")
    print(pd.DataFrame(rows).set_index(["model", "split"]).round(4).to_string())
    print("\nBaseline to beat, tuned Elo on validation: log loss 0.6105, Brier 0.2117, accuracy 0.6549")

    full_set = "elo + form + rest"
    print(f"\nStandardized weights, logistic: {full_set}")
    print("(positive = favors the home team; size = change in log-odds per one standard deviation)")
    weights = standardized_weights(models[full_set])
    print(weights.round(3).to_string())
    home_edge = 1 / (1 + np.exp(-weights["home_court"]))
    print(f"\nLearned home-court advantage: equal teams -> home wins {home_edge:.1%} (neutral site: 50.0%)")
