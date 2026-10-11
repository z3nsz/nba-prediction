"""Leakage tests for probability-of-playing and player-impact features (src/injury_impact.py)."""

import pandas as pd
import pytest

from src.injury_impact import PRIOR_WEIGHT, play_probabilities, player_value_table


def test_play_probability_uses_only_earlier_listings():
    """One player listed Questionable on three dates: played, did not play, then the game being predicted."""
    players = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-11-01", "2024-11-03", "2024-11-05"]),
        "name_key": ["someplayer"] * 3,
        "status": ["Questionable"] * 3,
        "played": [True, False, True],  # the Nov 5 outcome must not affect the Nov 5 probability
    })
    p = play_probabilities(players).to_numpy()
    assert p[0] == pytest.approx(0.5), "no history at all: neutral prior"
    # Nov 3: one earlier listing (played). Status rate = (1+1)/(1+2); own history 1 of 1.
    status_rate = 2 / 3
    assert p[1] == pytest.approx((1 + PRIOR_WEIGHT * status_rate) / (1 + PRIOR_WEIGHT))
    # Nov 5 uses Nov 1 and Nov 3 only. Changing the Nov 5 result must not change it.
    flipped = players.assign(played=[True, False, False])
    assert play_probabilities(flipped).to_numpy()[2] == pytest.approx(p[2])


def test_player_value_shrinks_toward_prior_and_uses_box_scores_up_to_each_date():
    logs = pd.DataFrame({
        "season": ["2024-25"] * 2, "name_key": ["star", "bench"], "GAME_DATE": pd.to_datetime(["2024-11-01"] * 2),
        "MIN": [36, 12], "PTS": [40, 2], "FGM": [0, 0], "FGA": [0, 0], "FTA": [0, 0], "FTM": [0, 0],
        "OREB": [0, 0], "DREB": [0, 0], "STL": [0, 0], "AST": [0, 0], "BLK": [0, 0], "PF": [0, 0], "TOV": [0, 0],
    })
    values = player_value_table(logs).set_index("name_key")
    assert values.loc["star", "rate_above_league"] > 0 > values.loc["bench", "rate_above_league"]
    # One game is weak evidence: the shrunk rate is much closer to the league rate than the raw 40/36.
    raw_star = 40 / 36
    league = values.loc["star", "league_rate"]
    assert abs(values.loc["star", "rate_above_league"]) < abs(raw_star - league) * 0.2
