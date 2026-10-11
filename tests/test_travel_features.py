"""Tests for travel and fatigue features (src/travel_features.py)."""

import pandas as pd
import pytest

from src.travel_features import ARENAS, distance_km, game_venues, travel_features, utc_offset_hours


def test_distance_between_new_york_and_los_angeles():
    nyc, la = ARENAS["NYK"], ARENAS["LAL"]
    assert distance_km(nyc[0], nyc[1], la[0], la[1]) == pytest.approx(3940, abs=40)


def test_arizona_daylight_saving_is_handled():
    """Phoenix matches Denver in winter but Los Angeles in late spring (it never changes its clocks)."""
    winter, spring = pd.Timestamp("2024-01-15"), pd.Timestamp("2024-04-10")
    assert utc_offset_hours("America/Phoenix", winter) == utc_offset_hours("America/Denver", winter)
    assert utc_offset_hours("America/Phoenix", spring) == utc_offset_hours("America/Los_Angeles", spring)


def test_every_neutral_site_game_has_a_venue(games):
    venues = game_venues(games)
    assert not venues[games["is_neutral_site"] == 1].isin(ARENAS).any()


def test_toronto_tampa_home_games_are_home_games(games):
    """2020-21 Raptors home games were in Tampa, but they were not road games."""
    travel = travel_features(games).merge(games[["game_id", "season", "home_team"]], on="game_id")
    tampa = travel[(travel["season"] == "2020-21") & (travel["home_team"] == "TOR") & (travel["team"] == "TOR")]
    assert len(tampa) == 36
    assert (tampa["road_trip_game"] == 0).all()


def hand_made_schedule() -> pd.DataFrame:
    """Team AAA: away at BBB, away at CCC, home vs DDD, all in one season."""
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "game_date": pd.to_datetime(["2024-11-01", "2024-11-02", "2024-11-05"]),
        "season": ["2024-25"] * 3,
        "home_team": ["BOS", "NYK", "LAL"],
        "away_team": ["LAL", "LAL", "MIA"],
        "is_neutral_site": [0, 0, 0],
    })


def test_road_trip_counter_and_travel_on_hand_made_schedule():
    travel = travel_features(hand_made_schedule()).set_index(["game_id", "team"])
    assert travel.loc[("g1", "LAL"), "road_trip_game"] == 1
    assert travel.loc[("g2", "LAL"), "road_trip_game"] == 2
    assert travel.loc[("g3", "LAL"), "road_trip_game"] == 0, "back home resets the road trip"
    assert pd.isna(travel.loc[("g1", "LAL"), "travel_km"]), "season opener: no previous venue"
    assert travel.loc[("g2", "LAL"), "travel_km"] == pytest.approx(distance_km(*ARENAS["BOS"][:2], *ARENAS["NYK"][:2]))
    assert travel.loc[("g3", "LAL"), "tz_change_hours"] == -3, "New York to Los Angeles = 3 hours west"
