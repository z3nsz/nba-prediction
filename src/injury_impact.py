"""Experiments 1 and 2: probability of playing, and player impact, for injury-listed players.

Experiment 1: probability of playing
    For each status (Out, Doubtful, Questionable, Probable, Available) and each
    player, how often did listed players ACTUALLY play in EARLIER games? Whether
    an earlier listed player played comes from that earlier game's box score,
    which was public long before the game being predicted. The predicted game's
    own box score is never used.

        P(play) = (player's earlier plays + PRIOR_WEIGHT * status rate) / (player's earlier listings + PRIOR_WEIGHT)
        expected missing minutes = average minutes * (1 - P(play))

    "status rate" is the share of all earlier listings with that status that played,
    starting from a neutral Beta(1, 1) prior (= 50%) before any history exists.

Experiment 2: player impact relative to an average player
    rate      = Game Score per minute this season, from earlier dates only, shrunk toward
                last season's rate (or the league rate) with SHRINK_MINUTES of pseudo-minutes
    value     = (rate - league rate) * average minutes   (0 = average player, stars >> 0)
    team feature = sum over listed players of value * (1 - P(play))
    stars     = expected number of missing players averaging 18+ Game Score (all-star level)

Usage:
    python -m src.injury_impact
"""

import numpy as np
import pandas as pd

from src.availability_features import (AVAILABILITY_PATH, MANIFEST_PATH, game_score, listed_players,
                                       listed_rows, load_player_logs)
from src.clean_data import PROCESSED_DATA_DIR, load_games_table
from src.walk_forward import DEVELOPMENT_SEASONS

IMPACT_PATH = PROCESSED_DATA_DIR / "injury_impact_features.csv"
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
PRIOR_WEIGHT = 5       # how many "average listings" a player's own history must outweigh
SHRINK_MINUTES = 300   # pseudo-minutes pulling an early-season rate toward the prior
STAR_GAME_SCORE = 18
IMPACT_COLUMNS = ["exp_missing_minutes", "exp_missing_q_minutes", "exp_missing_value", "exp_stars_missing"]


def played_flags(players: pd.DataFrame, logs: pd.DataFrame) -> pd.Series:
    """For each listed player-game: did he appear in that game's box score? (used ONLY as a label for later games)"""
    appeared = set(zip(logs["season"], logs["name_key"], logs["GAME_DATE"]))
    return pd.Series([(s, k, d) in appeared for s, k, d in
                      zip(players["season"], players["name_key"], players["game_date"])], index=players.index)


def play_probabilities(players: pd.DataFrame) -> pd.Series:
    """P(play) for each listed player-game, using only listings on strictly earlier dates."""
    rows = players[["game_date", "name_key", "status", "played"]].copy()

    # Status-wide rate from all earlier dates: cumulative counts per status, shifted by one date.
    daily = rows.groupby(["status", "game_date"])["played"].agg(["sum", "size"]).reset_index().sort_values("game_date")
    by_status = daily.groupby("status")
    daily["plays_before"] = by_status["sum"].cumsum() - daily["sum"]
    daily["listings_before"] = by_status["size"].cumsum() - daily["size"]
    daily["status_rate"] = (daily["plays_before"] + 1) / (daily["listings_before"] + 2)  # Beta(1, 1) prior
    rows = rows.merge(daily[["status", "game_date", "status_rate"]], on=["status", "game_date"], how="left")

    # The player's own earlier listings with the same status (all seasons so far), shrunk toward the status rate.
    per_player = (rows.groupby(["name_key", "status", "game_date"])["played"].agg(["sum", "size"])
                  .reset_index().sort_values("game_date"))
    grouped = per_player.groupby(["name_key", "status"])
    per_player["own_plays"] = grouped["sum"].cumsum() - per_player["sum"]
    per_player["own_listings"] = grouped["size"].cumsum() - per_player["size"]
    rows = rows.merge(per_player[["name_key", "status", "game_date", "own_plays", "own_listings"]],
                      on=["name_key", "status", "game_date"], how="left")
    p_play = ((rows["own_plays"] + PRIOR_WEIGHT * rows["status_rate"]) / (rows["own_listings"] + PRIOR_WEIGHT))
    return pd.Series(p_play.to_numpy(), index=players.index)


def player_value_table(logs: pd.DataFrame) -> pd.DataFrame:
    """Shrunk Game Score per minute (above league average) after each date a player played."""
    logs = logs.assign(GAME_SCORE=game_score(logs)).sort_values("GAME_DATE")
    # League rate from all EARLIER dates (cumulative, excluding the same date).
    league_daily = logs.groupby("GAME_DATE")[["GAME_SCORE", "MIN"]].sum().cumsum()
    league_rate = (league_daily["GAME_SCORE"] / league_daily["MIN"]).rename("league_rate")

    days = (logs.groupby(["season", "name_key", "GAME_DATE"], as_index=False)[["GAME_SCORE", "MIN"]].sum()
            .sort_values("GAME_DATE"))
    grouped = days.groupby(["season", "name_key"])
    days["gs_sum"], days["min_sum"] = grouped["GAME_SCORE"].cumsum(), grouped["MIN"].cumsum()
    days["league_rate"] = days["GAME_DATE"].map(league_rate)

    # Prior: the player's final rate from the previous season, if he played; else the league rate.
    final = days.groupby(["season", "name_key"]).last()
    final_rate = (final["gs_sum"] / final["min_sum"].replace(0, np.nan)).rename("prior_rate").reset_index()
    seasons = sorted(days["season"].unique())
    final_rate["season"] = final_rate["season"].map(dict(zip(seasons[:-1], seasons[1:])))
    days = days.merge(final_rate.dropna(subset=["season"]), on=["season", "name_key"], how="left")
    prior = days["prior_rate"].fillna(days["league_rate"])
    days["rate"] = (days["gs_sum"] + SHRINK_MINUTES * prior) / (days["min_sum"] + SHRINK_MINUTES)
    days["rate_above_league"] = days["rate"] - days["league_rate"]
    return days[["season", "name_key", "GAME_DATE", "rate_above_league", "league_rate"]]


def injury_impact_features(games: pd.DataFrame, manifest: pd.DataFrame, logs: pd.DataFrame,
                           seasons=DEVELOPMENT_SEASONS, availability: pd.DataFrame | None = None):
    """Features for games in `seasons`. Histories (P(play), prior-season value) use every earlier date given."""
    games = games[games["season"].isin(seasons)]
    players = listed_players(listed_rows(games, manifest, seasons), logs)
    players = players[players["status"].isin(STATUSES)].copy()
    players["played"] = played_flags(players, logs)
    players["p_play"] = play_probabilities(players)

    values = player_value_table(logs).sort_values("GAME_DATE")
    players = pd.merge_asof(players.sort_values("game_date"), values, left_on="game_date", right_on="GAME_DATE",
                            by=["season", "name_key"], allow_exact_matches=False, direction="backward")
    players["rate_above_league"] = players["rate_above_league"].fillna(0)  # no games yet: treat as average

    miss = 1 - players["p_play"]
    players["exp_missing_minutes"] = players["avg_minutes"] * miss
    players["exp_missing_q_minutes"] = np.where(players["status"] == "Questionable", players["exp_missing_minutes"], 0)
    players["exp_missing_value"] = players["rate_above_league"] * players["avg_minutes"] * miss
    players["exp_stars_missing"] = (players["avg_game_score"] >= STAR_GAME_SCORE) * miss
    team = players.groupby(["game_id", "team_abbr"])[IMPACT_COLUMNS].sum().reset_index()

    # Same "known" rule as the baseline features: unknown report -> NaN.
    baseline = availability if availability is not None else pd.read_csv(AVAILABILITY_PATH, dtype={"game_id": str})
    result = games[["game_id"]].copy()
    for side in ["home", "away"]:
        side_values = (games[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team_abbr"})
                       .merge(team, on=["game_id", "team_abbr"], how="left"))
        known = side_values["game_id"].map(baseline.set_index("game_id")[f"{side}_report_found"]).eq(1)
        for column in IMPACT_COLUMNS:
            side_values[column] = side_values[column].fillna(0).where(known.to_numpy())
        result = result.merge(side_values.drop(columns="team_abbr")
                              .rename(columns={c: f"{side}_{c}" for c in IMPACT_COLUMNS}), on="game_id")
    return result, players


TEST_IMPACT_PATH = PROCESSED_DATA_DIR / "injury_impact_features_test.csv"

if __name__ == "__main__":
    import argparse
    from src.availability_features import TEST_AVAILABILITY_PATH
    from src.injury_download import TEST_MANIFEST_PATH
    from src.walk_forward import TEST_SEASONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-season", action="store_true", help="build 2025-26 features (protocol committed)")
    args = parser.parse_args()
    read = lambda path: pd.read_csv(path, dtype={"game_id": str}, parse_dates=["game_date", "tip_off"])
    if args.test_season:
        # Process development + test seasons together so histories reach back across seasons;
        # keep only test-season rows. Development rows are re-checked against the saved file.
        seasons = DEVELOPMENT_SEASONS + TEST_SEASONS
        manifest = pd.concat([read(MANIFEST_PATH), read(TEST_MANIFEST_PATH)], ignore_index=True)
        availability = pd.concat([pd.read_csv(AVAILABILITY_PATH, dtype={"game_id": str}),
                                  pd.read_csv(TEST_AVAILABILITY_PATH, dtype={"game_id": str})], ignore_index=True)
        table, players = injury_impact_features(load_games_table(), manifest, load_player_logs(seasons),
                                                seasons, availability)
        saved_dev = pd.read_csv(IMPACT_PATH, dtype={"game_id": str})
        rebuilt_dev = table[table["game_id"].isin(saved_dev["game_id"])].sort_values("game_id").reset_index(drop=True)
        saved_dev = saved_dev.sort_values("game_id").reset_index(drop=True)[rebuilt_dev.columns]
        values = [c for c in rebuilt_dev.columns if c != "game_id"]
        # Floating-point sums in a different order differ around 1e-14; anything above 1e-9 would be real.
        same = (rebuilt_dev["game_id"].equals(saved_dev["game_id"])
                and (rebuilt_dev[values].isna() == saved_dev[values].isna()).all().all()
                and np.allclose(rebuilt_dev[values].fillna(0), saved_dev[values].fillna(0), rtol=0, atol=1e-9))
        print(f"Development-season features unchanged after adding 2025-26 data (within 1e-9): {same}")
        if not same:
            raise SystemExit("Leakage check failed: adding later data changed earlier features.")
        test_games = load_games_table().query("season in @TEST_SEASONS")["game_id"]
        table[table["game_id"].isin(test_games)].to_csv(TEST_IMPACT_PATH, index=False)
        print(f"Saved {table['game_id'].isin(test_games).sum()} test-season games -> {TEST_IMPACT_PATH.name}")
        raise SystemExit
    manifest = read(MANIFEST_PATH)
    table, players = injury_impact_features(load_games_table(), manifest, load_player_logs())
    table.to_csv(IMPACT_PATH, index=False)
    print(f"Saved {len(table)} games -> data/processed/{IMPACT_PATH.name}\n")
    print("How often listed players actually played (all development seasons):")
    print(players.groupby("status")["played"].agg(["mean", "size"]).round(3).to_string())
    print("\nAverage predicted P(play) vs. actual, by status:")
    print(players.groupby("status")[["p_play", "played"]].mean().round(3).to_string())
