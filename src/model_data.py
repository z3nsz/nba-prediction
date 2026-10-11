"""Turn the features table into model inputs (X) and the target (y).

Each input is a home-minus-away difference (or the home-court flag), so a
positive value always means "this favors the home team". That keeps the
number of inputs small and makes model coefficients easy to read.

Season openers have no rolling history (NaN). A missing difference is filled
with 0, meaning "no evidence that either team is better on this measure".
That is a fixed constant, not information from other games, so it cannot leak.
"""

import pandas as pd

from src.evaluate import TARGET


def make_model_inputs(features: pd.DataFrame) -> pd.DataFrame:
    """Build every candidate model input from the features table."""
    inputs = pd.DataFrame(index=features.index)
    inputs["elo_diff"] = features["elo_diff"]
    inputs["home_court"] = features["home_court"]
    inputs["point_diff_last10_diff"] = features["home_avg_point_diff_last10"] - features["away_avg_point_diff_last10"]
    inputs["win_pct_last10_diff"] = features["home_win_pct_last10"] - features["away_win_pct_last10"]
    inputs["win_pct_last5_diff"] = features["home_win_pct_last5"] - features["away_win_pct_last5"]
    inputs["rest_days_diff"] = features["home_days_since_last_game"] - features["away_days_since_last_game"]
    # Positive = the AWAY team is the one on a back-to-back (good for the home team).
    inputs["back_to_back_diff"] = features["away_is_back_to_back"] - features["home_is_back_to_back"]

    # Phase 3 strength features (src/strength_features.py).
    for window in [5, 10]:
        inputs[f"net_rating_last{window}_diff"] = (features[f"home_net_rating_last{window}"]
                                                   - features[f"away_net_rating_last{window}"])
        inputs[f"off_rating_last{window}_diff"] = (features[f"home_off_rating_last{window}"]
                                                   - features[f"away_off_rating_last{window}"])
        # Defensive rating: lower is better, so away minus home keeps "positive favors home".
        inputs[f"def_rating_last{window}_diff"] = (features[f"away_def_rating_last{window}"]
                                                   - features[f"home_def_rating_last{window}"])
        # Positive = the home team's recent opponents were stronger.
        inputs[f"opp_elo_last{window}_diff"] = features[f"home_opp_elo_last{window}"] - features[f"away_opp_elo_last{window}"]
    inputs["margin_vs_elo_last10_diff"] = features["home_margin_vs_elo_last10"] - features["away_margin_vs_elo_last10"]

    # Phase 3 travel features (src/travel_features.py). Away minus home: positive = the away
    # team traveled more / is deeper into a road trip / crossed more time zones.
    inputs["travel_km_diff"] = features["away_travel_km"] - features["home_travel_km"]
    inputs["travel_km_last7d_diff"] = features["away_travel_km_last7d"] - features["home_travel_km_last7d"]
    inputs["tz_change_abs_diff"] = features["away_tz_change_hours"].abs() - features["home_tz_change_hours"].abs()
    inputs["road_trip_game_diff"] = features["away_road_trip_game"] - features["home_road_trip_game"]

    # Phase 3 availability features (src/availability_features.py), present only when merged in.
    # Away minus home: positive = the AWAY team is missing more (good for the home team).
    # Unknown (no usable report) becomes 0 below = "no evidence either team is missing more".
    if "home_out_minutes" in features:
        for stat in ["out_minutes", "questionable_minutes", "starters_out",
                     "out_game_score", "questionable_game_score", "out_plus_minus"]:
            if f"home_{stat}" in features:
                inputs[f"{stat}_diff"] = features[f"away_{stat}"] - features[f"home_{stat}"]
    # Accuracy experiments (src/injury_impact.py): probability-weighted missing minutes and player impact.
    for stat in ["exp_missing_minutes", "exp_missing_q_minutes", "exp_missing_value", "exp_stars_missing"]:
        if f"home_{stat}" in features:
            inputs[f"{stat}_diff"] = features[f"away_{stat}"] - features[f"home_{stat}"]
    return inputs.fillna(0)


# Groups of inputs, built up one idea at a time so we can see what each adds.
FEATURE_SETS = {
    "elo only": ["elo_diff", "home_court"],
    "elo + form": ["elo_diff", "home_court", "point_diff_last10_diff", "win_pct_last10_diff", "win_pct_last5_diff"],
    "elo + form + rest": ["elo_diff", "home_court", "point_diff_last10_diff", "win_pct_last10_diff",
                          "win_pct_last5_diff", "rest_days_diff", "back_to_back_diff"],
}


# Phase 3 feature groups, added one at a time on top of the Phase 2 inputs (ablation, Step 3.3).
PHASE2_INPUTS = FEATURE_SETS["elo + form + rest"]
FEATURE_GROUPS = {
    # Offense and defense separately. Net rating is NOT added on top: net_rating_diff equals
    # off_rating_diff + def_rating_diff exactly, so it would repeat the same information.
    "ratings": [f"{stat}_last{window}_diff" for window in [5, 10] for stat in ["off_rating", "def_rating"]],
    "schedule strength": ["opp_elo_last5_diff", "opp_elo_last10_diff", "margin_vs_elo_last10_diff"],
    "availability": ["out_minutes_diff", "questionable_minutes_diff", "starters_out_diff"],
    "player impact": ["out_game_score_diff", "questionable_game_score_diff"],
    "plus-minus impact": ["out_plus_minus_diff"],
    "travel": ["travel_km_diff", "travel_km_last7d_diff", "tz_change_abs_diff", "road_trip_game_diff"],
}


def x_and_y(split: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs restricted to `columns`, and the target, for one split."""
    return make_model_inputs(split)[columns], split[TARGET]
