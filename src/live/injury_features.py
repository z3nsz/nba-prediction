"""Live injury inputs for the model, computed by the SAME functions as training (src/injury_impact.py).

Inputs (per team, then away minus home):
    exp_missing_minutes  listed players' season average minutes x chance they miss the game
    starters_out         Out/Doubtful players averaging 28+ minutes
    exp_missing_value    Game Score per minute above league average x minutes x chance they miss
    exp_stars_missing    expected number of missing players averaging 18+ Game Score

Training used six seasons of raw box scores and listings. The live job instead carries a small STATE that
summarises the past, and turns it back into a few synthetic rows the training functions can read:
    value_state.csv    league Game Score and minutes totals since 2020-21 (one "league" row) and each
                       player's totals for the last finished season (his prior rate)
    listing_state.csv  for each player and status: times listed, times he then played
Current-season box scores and listings are added as they happen.

Player names: ESPN drops suffixes ("Bobby Portis") that injury reports keep ("Portis Jr., Bobby"),
so live matching uses a key that ignores Jr./Sr./II/III/IV.

Usage:
    python -m src.live.injury_features        (build the state from all six seasons)
"""

import re

import numpy as np
import pandas as pd

from src.availability_features import STARTER_MINUTES, UNAVAILABLE, attach_expected_minutes, expected_minutes_table
from src.injury_impact import STAR_GAME_SCORE, STATUSES, play_probabilities, player_value_table
from src.injury_reports import normalize_player_name
from src.live.team_features import STATE_DIR

VALUE_STATE_PATH = STATE_DIR / "value_state.csv"
LISTING_STATE_PATH = STATE_DIR / "listing_state.csv"
BOX_COLUMNS = ["MIN", "PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
LEAGUE_KEY = "__league__"
HISTORY_SEASON = "0000"   # sorts before every real season
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.IGNORECASE)
INJURY_INPUTS = ["exp_missing_minutes", "starters_out", "exp_missing_value", "exp_stars_missing"]


def live_key(name: str) -> str:
    """Name key that ignores suffixes: 'Portis Jr., Bobby' and 'Bobby Portis' -> 'bobbyportis'."""
    return normalize_player_name(SUFFIX.sub(" ", str(name)))


def to_logs(box: pd.DataFrame, season: str) -> pd.DataFrame:
    """Box-score rows (player, date, stats) in the player-log format the training functions expect."""
    key = box["player"].map(live_key)
    logs = box[["game_date", *BOX_COLUMNS]].rename(columns={"game_date": "GAME_DATE"}).copy()
    return logs.assign(season=season, name_key=key, PLAYER_ID=key)


# ---------------------------------------------------------------- state (summary of finished seasons)

def build_state(logs: pd.DataFrame, listings: pd.DataFrame, last_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """logs: player logs for every finished season (training format, live keys); listings: past listings
    (name_key, status, played). Returns (value_state, listing_state)."""
    logs = logs.assign(GAME_SCORE=player_game_score(logs))
    before = logs[logs["season"] < last_season]
    league = pd.DataFrame([{"name_key": LEAGUE_KEY, "gs_sum": before["GAME_SCORE"].sum(), "min_sum": before["MIN"].sum()}])
    players = logs[logs["season"] == last_season].groupby("name_key")[["GAME_SCORE", "MIN"]].sum().reset_index()
    players = players.rename(columns={"GAME_SCORE": "gs_sum", "MIN": "min_sum"})
    value_state = pd.concat([league, players], ignore_index=True).assign(season=last_season)
    listing_state = (listings.groupby(["name_key", "status"])["played"].agg(plays="sum", listings="size").reset_index())
    return value_state, listing_state


def player_game_score(logs: pd.DataFrame) -> pd.Series:
    from src.availability_features import game_score
    return game_score(logs)


def history_logs(value_state: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Synthetic log rows that reproduce the past: league totals and each player's last-season totals.

    A row with PTS = total Game Score and every other stat 0 has Game Score = PTS, so the training functions'
    running sums start from exactly the historical totals.
    """
    last_season = value_state["season"].iloc[0]
    rows = value_state.assign(
        season=np.where(value_state["name_key"] == LEAGUE_KEY, HISTORY_SEASON, last_season),
        GAME_DATE=before - pd.Timedelta(days=1), PLAYER_ID=value_state["name_key"],
        PTS=value_state["gs_sum"], MIN=value_state["min_sum"])
    for column in BOX_COLUMNS:
        if column not in ("PTS", "MIN"):
            rows[column] = 0
    return rows[["season", "PLAYER_ID", "name_key", "GAME_DATE", *BOX_COLUMNS]]


def history_listings(listing_state: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Expand (player, status, plays, listings) back into one row per past listing, dated before the season."""
    rows = listing_state.loc[listing_state.index.repeat(listing_state["listings"])].copy()
    rows["played"] = rows.groupby(["name_key", "status"]).cumcount() < rows["plays"]
    return rows.assign(game_date=before - pd.Timedelta(days=1))[["game_date", "name_key", "status", "played"]]


# ---------------------------------------------------------------- the features

def injury_inputs(day: pd.Timestamp, season: str, listed: pd.DataFrame, season_logs: pd.DataFrame,
                  season_listings: pd.DataFrame, value_state: pd.DataFrame, listing_state: pd.DataFrame) -> pd.DataFrame:
    """Per-team injury inputs for `day`'s games.

    listed:          today's listed players: game_id, team_abbr, player, status
    season_logs:     this season's player logs before `day` (training format, live keys)
    season_listings: this season's listings before `day`: game_date, name_key, status, played
    """
    season_start = min([day, *pd.to_datetime(season_logs["GAME_DATE"]).tolist(),
                        *pd.to_datetime(season_listings["game_date"]).tolist()])
    logs = pd.concat([history_logs(value_state, season_start), season_logs[season_logs["GAME_DATE"] < day]],
                     ignore_index=True)

    players = listed[listed["status"].isin(STATUSES)].copy()
    players = players.assign(name_key=players["player"].map(live_key), game_date=day, season=season)
    if players.empty:
        return pd.DataFrame(columns=["game_id", "team_abbr", *INJURY_INPUTS])
    players = attach_expected_minutes(players, expected_minutes_table(logs)).reset_index(drop=True)
    players[["avg_minutes", "avg_game_score"]] = players[["avg_minutes", "avg_game_score"]].fillna(0)

    past = pd.concat([history_listings(listing_state, season_start),
                      season_listings[season_listings["game_date"] < day]], ignore_index=True)
    combined = pd.concat([past, players[["game_date", "name_key", "status"]].assign(played=False)], ignore_index=True)
    # Set types explicitly: an empty season log (opening night) would otherwise make these "object".
    combined["played"] = combined["played"].astype(bool)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    players["p_play"] = play_probabilities(combined).iloc[len(past):].to_numpy()

    values = player_value_table(logs).sort_values("GAME_DATE")
    players = pd.merge_asof(players.sort_values("game_date"), values, left_on="game_date", right_on="GAME_DATE",
                            by=["season", "name_key"], allow_exact_matches=False, direction="backward")
    players["rate_above_league"] = players["rate_above_league"].fillna(0)

    miss = 1 - players["p_play"]
    is_out = players["status"].isin(UNAVAILABLE)
    players["exp_missing_minutes"] = players["avg_minutes"] * miss
    players["starters_out"] = (is_out & (players["avg_minutes"] >= STARTER_MINUTES)).astype(int)
    players["exp_missing_value"] = players["rate_above_league"] * players["avg_minutes"] * miss
    players["exp_stars_missing"] = (players["avg_game_score"] >= STAR_GAME_SCORE) * miss
    return players.groupby(["game_id", "team_abbr"])[INJURY_INPUTS].sum().reset_index()


def injury_diffs(games: pd.DataFrame, per_team: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Away minus home per input (positive favors home). Unknown reports give 0, as in training.

    games: game_id, home_team, away_team; known: game_id, team_abbr, known (bool).
    """
    out = games[["game_id"]].reset_index(drop=True)
    for side in ["home", "away"]:
        teams = games[["game_id", f"{side}_team"]].rename(columns={f"{side}_team": "team_abbr"}).reset_index(drop=True)
        values = teams.merge(per_team, on=["game_id", "team_abbr"], how="left").fillna({c: 0 for c in INJURY_INPUTS})
        is_known = teams.merge(known, on=["game_id", "team_abbr"], how="left")["known"].fillna(False).astype(bool)
        for column in INJURY_INPUTS:
            # Plain arrays: assigning a Series would align on index labels, not row positions.
            out[f"{side}_{column}"] = values[column].where(is_known).to_numpy()
    for column in INJURY_INPUTS:
        out[f"{column}_diff"] = (out[f"away_{column}"] - out[f"home_{column}"]).fillna(0)
    return out


if __name__ == "__main__":
    from src.live.state_builder import write_state
    write_state()
