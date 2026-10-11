"""Safeguard tests for the one-time final test (src/final_test.py). These never score the test season."""

import json
import subprocess
import sys

import pytest

from src.final_test import RESULTS_PATH, SETTINGS_PATH
from src.walk_forward import DEVELOPMENT_SEASONS, TEST_SEASONS


def test_refuses_to_run_without_confirmation():
    result = subprocess.run([sys.executable, "-m", "src.final_test"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to run" in result.stderr


def test_refuses_a_second_run_once_results_exist():
    """The one-time test has been run; even WITH the confirmation flag it must refuse now."""
    if not RESULTS_PATH.exists():
        pytest.skip("final test not run yet; running it here would use up the test season")
    before = RESULTS_PATH.read_text()
    result = subprocess.run([sys.executable, "-m", "src.final_test", "--confirm-one-time-test"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert RESULTS_PATH.read_text() == before, "results must not be overwritten"


def test_locked_settings_train_only_on_development_seasons():
    settings = json.loads(SETTINGS_PATH.read_text())
    assert settings["final_model"]["train_seasons"] == DEVELOPMENT_SEASONS
    assert settings["test_season"] in TEST_SEASONS
    assert settings["final_model"]["inputs"][:7] == [
        "elo_diff", "home_court", "point_diff_last10_diff", "win_pct_last10_diff",
        "win_pct_last5_diff", "rest_days_diff", "back_to_back_diff"]


def test_locked_model_is_exactly_the_evaluated_e1_e2_model():
    from src.accuracy_experiments import BASELINE_INPUTS, EXPERIMENTS
    settings = json.loads(SETTINGS_PATH.read_text())
    assert settings["final_model"]["inputs"] == EXPERIMENTS["E1+E2: P(miss) minutes + value + stars"]
    assert settings["comparison_models"]["logistic_10_inputs"]["inputs"] == BASELINE_INPUTS
    assert settings["final_model"]["c"] == 1.0
