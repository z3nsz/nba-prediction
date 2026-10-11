"""The one-time final evaluation on the 2025-26 test season (see docs/final_test_protocol.md).

Safeguards:
    - refuses to run without --confirm-one-time-test
    - refuses to run if results/final_test_results.json already exists (no second tries)
    - every setting is read from results/final_model_settings.json, written before the test

Usage (once, after the protocol is committed and 2025-26 availability features are built):
    python -m src.final_test --confirm-one-time-test
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.availability_features import AVAILABILITY_PATH
from src.clean_data import PROCESSED_DATA_DIR, load_games_table
from src.compare_models import calibration_bins, expected_calibration_error
from src.elo_baseline import elo_home_win_probs
from src.evaluate import TARGET, score_predictions
from src.feature_engineering import load_features_table
from src.injury_impact import IMPACT_PATH
from src.logistic_model import predict_home_win, train_logistic
from src.tuning import RESULTS_DIR
from src.walk_forward import date_block_bootstrap

SETTINGS_PATH = RESULTS_DIR / "final_model_settings.json"
RESULTS_PATH = RESULTS_DIR / "final_test_results.json"
PREDICTIONS_PATH = RESULTS_DIR / "final_test_predictions.csv"
TEST_AVAILABILITY_PATH = PROCESSED_DATA_DIR / "availability_features_test.csv"
TEST_IMPACT_PATH = PROCESSED_DATA_DIR / "injury_impact_features_test.csv"
PRIMARY = "E1+E2 (final model)"
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]


def check_safeguards(args) -> None:
    if not args.confirm_one_time_test:
        sys.exit("Refusing to run: this evaluates the untouched test season ONCE. "
                 "Re-run with --confirm-one-time-test if the protocol is committed.")
    if RESULTS_PATH.exists():
        sys.exit(f"Refusing to run: {RESULTS_PATH.name} already exists. The test season can only be used once.")
    for path in [TEST_AVAILABILITY_PATH, TEST_IMPACT_PATH]:
        if not path.exists():
            sys.exit(f"Missing {path.name}: build 2025-26 injury features first.")


def load_inputs(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability = pd.concat([pd.read_csv(AVAILABILITY_PATH, dtype={"game_id": str}),
                              pd.read_csv(TEST_AVAILABILITY_PATH, dtype={"game_id": str})], ignore_index=True)
    impact = pd.concat([pd.read_csv(IMPACT_PATH, dtype={"game_id": str}),
                        pd.read_csv(TEST_IMPACT_PATH, dtype={"game_id": str})], ignore_index=True)
    features = (load_features_table().merge(availability, on="game_id", how="left", validate="one_to_one")
                .merge(impact, on="game_id", how="left", validate="one_to_one"))
    train = features[features["season"].isin(settings["final_model"]["train_seasons"])]
    test = features[features["season"] == settings["test_season"]]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-one-time-test", action="store_true")
    args = parser.parse_args()
    check_safeguards(args)

    settings = json.loads(SETTINGS_PATH.read_text())
    train, test = load_inputs(settings)
    inputs = settings["final_model"]["inputs"]
    c = settings["final_model"]["c"]
    elo = settings["comparison_models"]["elo_tuned"]

    predictions = test[["game_id", "game_date", TARGET]].copy()
    predictions[PRIMARY] = predict_home_win(train_logistic(train, inputs, c=c), test, inputs)
    ten_inputs = settings["comparison_models"]["logistic_10_inputs"]["inputs"]
    predictions["logistic 10 inputs"] = predict_home_win(train_logistic(train, ten_inputs, c=c), test, ten_inputs)
    predictions["logistic phase 2"] = predict_home_win(train_logistic(train, inputs[:7], c=c), test, inputs[:7])
    predictions["Elo (tuned)"] = elo_home_win_probs(load_games_table(), k=elo["k"],
                                                    home_advantage=elo["home_advantage"]).loc[test["game_id"]].values
    predictions["always home"] = train[TARGET].mean()

    models = [PRIMARY, "logistic 10 inputs", "logistic phase 2", "Elo (tuned)", "always home"]
    scores = {m: {**score_predictions(predictions[TARGET], predictions[m]),
                  "calibration_error": expected_calibration_error(predictions[TARGET], predictions[m])}
              for m in models}
    gaps = {}
    for m in models[1:]:
        gap, low, high = date_block_bootstrap(predictions, PRIMARY, m)
        gaps[f"{PRIMARY} minus {m}"] = {"log_loss_gap": gap, "ci_low": low, "ci_high": high}

    confidence = np.maximum(predictions[PRIMARY], 1 - predictions[PRIMARY])
    correct = (predictions[PRIMARY] > 0.5) == predictions[TARGET].astype(bool)
    thresholds = {f"{t:.0%}": {"games_covered": float((confidence >= t).mean()),
                               "accuracy": float(correct[confidence >= t].mean())} for t in THRESHOLDS}
    results = {"games": len(predictions), "scores": scores, "log_loss_gaps": gaps, "confidence_thresholds": thresholds,
               "primary_calibration": calibration_bins(predictions[TARGET], predictions[PRIMARY]).round(4).to_dict("index")}
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=float) + "\n")
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Final test on {len(predictions)} games of {settings['test_season']}\n")
    print(pd.DataFrame(scores).T.round(4).to_string())
    print("\nLog loss gap (negative = primary model better), 95% date-block interval:")
    for name, g in gaps.items():
        print(f"  {name:45s} {g['log_loss_gap']:+.4f} [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]")
    print(f"\nSaved {RESULTS_PATH.name} and {PREDICTIONS_PATH.name}")


if __name__ == "__main__":
    main()
