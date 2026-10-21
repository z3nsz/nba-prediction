"""Build the live injury state from our NBA data, and check it by replaying 2025-26.

Runs locally (it needs the full NBA data), not in the cloud job.

    python -m src.live.state_builder            write live_state/value_state.csv + listing_state.csv (2020-26)
    python -m src.live.state_builder --replay   replay 2025-26 days with a state that ends in 2024-25 and
                                                compare with the original injury features for those games
"""

import argparse

import numpy as np
import pandas as pd

from src.availability_features import (AVAILABILITY_PATH, TEST_AVAILABILITY_PATH, listed_rows, load_player_logs)
from src.clean_data import load_games_table
from src.collect_data import SEASONS
from src.injury_download import MANIFEST_PATH, TEST_MANIFEST_PATH
from src.injury_impact import IMPACT_PATH, STATUSES, TEST_IMPACT_PATH, injury_impact_features
from src.live.injury_features import (INJURY_INPUTS, LISTING_STATE_PATH, VALUE_STATE_PATH, build_state,
                                      injury_diffs, injury_inputs, live_key)

read = lambda path: pd.read_csv(path, dtype={"game_id": str}, parse_dates=["game_date", "tip_off"])


def manifest_for(seasons) -> pd.DataFrame:
    paths = [MANIFEST_PATH] + ([TEST_MANIFEST_PATH] if "2025-26" in seasons else [])
    return pd.concat([read(path) for path in paths], ignore_index=True)


def availability_for(seasons) -> pd.DataFrame:
    paths = [AVAILABILITY_PATH] + ([TEST_AVAILABILITY_PATH] if "2025-26" in seasons else [])
    return pd.concat([pd.read_csv(p, dtype={"game_id": str}) for p in paths], ignore_index=True)


def logs_with_live_keys(seasons) -> pd.DataFrame:
    logs = load_player_logs(seasons)
    return logs.assign(name_key=logs["PLAYER_NAME"].map(live_key), PLAYER_ID=logs["PLAYER_NAME"].map(live_key))


def past_listings(seasons) -> pd.DataFrame:
    """Every listed player-game in `seasons` with whether he played (the training listing table)."""
    games = load_games_table()
    _, players = injury_impact_features(games, manifest_for(seasons), load_player_logs(seasons),
                                        list(seasons), availability_for(seasons))
    return players.assign(name_key=players["player"].map(live_key))[["game_date", "season", "name_key", "status",
                                                                       "played", "player", "game_id", "team_abbr"]]


def write_state(last_season: str = SEASONS[-1]) -> None:
    seasons = SEASONS[:SEASONS.index(last_season) + 1]
    value_state, listing_state = build_state(logs_with_live_keys(seasons), past_listings(seasons), last_season)
    VALUE_STATE_PATH.parent.mkdir(exist_ok=True)
    value_state.to_csv(VALUE_STATE_PATH, index=False)
    listing_state.to_csv(LISTING_STATE_PATH, index=False)
    print(f"value state: {len(value_state)} rows -> {VALUE_STATE_PATH.name} ({VALUE_STATE_PATH.stat().st_size / 1e3:.0f} KB)")
    print(f"listing state: {len(listing_state)} rows -> {LISTING_STATE_PATH.name} ({LISTING_STATE_PATH.stat().st_size / 1e3:.0f} KB)")


def replay(days: list[str]) -> pd.DataFrame:
    """State through 2024-25; 2025-26 data before each day; compare that day's games with the originals."""
    dev = SEASONS[:-1]
    value_state, listing_state = build_state(logs_with_live_keys(dev), past_listings(dev), dev[-1])
    test_logs = logs_with_live_keys(["2025-26"])
    test_listings = past_listings(SEASONS)        # includes 2025-26 listings with played flags
    test_listings = test_listings[test_listings["season"] == "2025-26"]
    games = load_games_table().query("season == '2025-26'")
    rows = listed_rows(games, read(TEST_MANIFEST_PATH), ["2025-26"])
    availability = pd.read_csv(TEST_AVAILABILITY_PATH, dtype={"game_id": str})
    original = (pd.read_csv(TEST_IMPACT_PATH, dtype={"game_id": str}).merge(availability, on="game_id"))

    results = []
    for date in days:
        day = pd.Timestamp(date)
        todays_games = games[games["game_date"] == day]
        listed = rows[rows["game_id"].isin(todays_games["game_id"]) & rows["player"].notna()]
        known = pd.concat([todays_games[["game_id", f"{s}_team"]].rename(columns={f"{s}_team": "team_abbr"})
                           .assign(known=todays_games["game_id"].map(availability.set_index("game_id")[f"{s}_report_found"]).eq(1).to_numpy())
                           for s in ["home", "away"]])
        per_team = injury_inputs(day, "2025-26", listed[["game_id", "team_abbr", "player", "status"]],
                                 test_logs[test_logs["GAME_DATE"] < day],
                                 test_listings[test_listings["game_date"] < day][["game_date", "name_key", "status", "played"]],
                                 value_state, listing_state)
        live = injury_diffs(todays_games, per_team, known).set_index("game_id")
        ref = original.set_index("game_id").loc[todays_games["game_id"]]
        source = {"exp_missing_minutes": "exp_missing_minutes", "starters_out": "starters_out",
                  "exp_missing_value": "exp_missing_value", "exp_stars_missing": "exp_stars_missing"}
        row = {"date": date, "games": len(todays_games)}
        for name, column in source.items():
            ref_diff = (ref[f"away_{column}"] - ref[f"home_{column}"]).fillna(0)
            row[name] = float((live[f"{name}_diff"] - ref_diff).abs().max())
        results.append(row)
    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        table = replay(["2025-10-23", "2025-11-14", "2025-12-26", "2026-01-15", "2026-02-05", "2026-03-10", "2026-04-12"])
        print("Largest absolute difference per input (live minus original), by replayed day:")
        print(table.to_string(index=False))
    else:
        write_state()
