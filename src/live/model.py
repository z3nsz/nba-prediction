"""The live model: the locked final model, retrained once on 2020-21 through 2025-26, stored as plain numbers.

Same inputs, settings and code as the pre-registered final model (results/final_model_settings.json),
fit on all six seasons. Saved as JSON (input names, scaling, weights) instead of a pickled scikit-learn
object, so the live job needs only numpy to predict and the file never breaks with library upgrades:

    P(home wins) = 1 / (1 + exp(-sum_i weight_i * input_i / scale_i))      (no intercept; home_court scale = 1)

Usage:
    python -m src.live.model          (retrain, save models/live_model.json, and verify it)
"""

import json
from datetime import date

import numpy as np
import pandas as pd

from src.collect_data import PROJECT_ROOT, SEASONS
from src.logistic_model import predict_home_win, train_logistic
from src.model_data import make_model_inputs

MODEL_PATH = PROJECT_ROOT / "models" / "live_model.json"
SETTINGS_PATH = PROJECT_ROOT / "results" / "final_model_settings.json"


def training_table() -> pd.DataFrame:
    """Features for all six seasons, with the injury features from the development and test builds."""
    from src.availability_features import AVAILABILITY_PATH, TEST_AVAILABILITY_PATH
    from src.feature_engineering import load_features_table
    from src.injury_impact import IMPACT_PATH, TEST_IMPACT_PATH
    read = lambda path: pd.read_csv(path, dtype={"game_id": str})
    availability = pd.concat([read(AVAILABILITY_PATH), read(TEST_AVAILABILITY_PATH)], ignore_index=True)
    impact = pd.concat([read(IMPACT_PATH), read(TEST_IMPACT_PATH)], ignore_index=True)
    features = (load_features_table().merge(availability, on="game_id", how="left", validate="one_to_one")
                .merge(impact, on="game_id", how="left", validate="one_to_one"))
    return features[features["season"].isin(SEASONS)]


def export(model, inputs: list[str], train_seasons: list[str], games: int) -> dict:
    scaler, logistic = model[0], model[-1]
    names = list(model[0].get_feature_names_out())
    scaled = list(scaler.named_transformers_["scale"].feature_names_in_)
    scales = dict(zip(scaled, scaler.named_transformers_["scale"].scale_))
    weights = dict(zip(names, logistic.coef_[0]))
    return {"description": "Final NBA model (logistic regression, no intercept), retrained on all six seasons",
            "trained_on": train_seasons, "training_games": games, "created": date.today().isoformat(),
            "inputs": inputs,
            "scale": {name: float(scales.get(name, 1.0)) for name in inputs},
            "weight": {name: float(weights[name]) for name in inputs}}


def load_model(path=MODEL_PATH) -> dict:
    return json.loads(path.read_text())


def predict(model: dict, inputs: pd.DataFrame) -> np.ndarray:
    """P(home wins) from the saved numbers: no scikit-learn needed."""
    z = sum(model["weight"][name] * inputs[name].to_numpy(dtype=float) / model["scale"][name]
            for name in model["inputs"])
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    settings = json.loads(SETTINGS_PATH.read_text())
    inputs, c = settings["final_model"]["inputs"], settings["final_model"]["c"]
    table = training_table()
    fitted = train_logistic(table, inputs, c=c)
    saved = export(fitted, inputs, SEASONS, len(table))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(saved, indent=2) + "\n")

    # Verify: the JSON model must give exactly the same probabilities as the scikit-learn model.
    from_json = predict(load_model(), make_model_inputs(table))
    from_sklearn = predict_home_win(fitted, table, inputs)
    print(f"Retrained on {len(table)} games ({SEASONS[0]} to {SEASONS[-1]}) -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"JSON vs scikit-learn predictions: max difference {np.abs(from_json - from_sklearn).max():.2e}")
    print("\nWeights (per standard deviation; home_court per game):")
    print(pd.Series(saved["weight"]).round(3).to_string())
