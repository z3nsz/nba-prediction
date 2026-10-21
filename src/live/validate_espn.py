"""Training/serving check: does ESPN's 2025-26 data match the stats.nba.com data the model was trained on?

Game level (every 2025-26 regular-season game): same games on the same dates, same final scores, same
neutral-site games, and tip-off times consistent with the official injury reports.
Feature level: rebuild the model's team features (Elo, form, rest) with ESPN scores in place of NBA scores
and compare with the original features.
Player level (random sample of games): same players, minutes and box-score stats, and whether ESPN player
names line up with the names used by our player logs (which the injury features rely on).

Usage:
    python -m src.live.validate_espn          (first run downloads ~270 ESPN responses, cached afterwards)
"""

import numpy as np
import pandas as pd

from src.availability_features import game_score
from src.clean_data import load_games_table
from src.collect_data import RAW_DATA_DIR
from src.feature_engineering import build_features_table
from src.injury_download import TEST_MANIFEST_PATH
from src.injury_reports import normalize_player_name
from src.live.espn import REGULAR_SEASON, box_score, scoreboard

SEASON = "2025-26"
PLAYER_SAMPLE_GAMES, SEED = 100, 3
TEAM_FEATURES = ["home_elo", "away_elo", "elo_diff", "home_win_pct_last10", "away_win_pct_last10",
                 "home_avg_point_diff_last10", "away_avg_point_diff_last10", "home_days_since_last_game",
                 "away_days_since_last_game", "home_is_back_to_back", "away_is_back_to_back", "home_court"]


def espn_season_games(dates: pd.Series) -> pd.DataFrame:
    frames = [scoreboard(pd.Timestamp(d)).assign(espn_date=pd.Timestamp(d)) for d in sorted(dates.unique())]
    espn = pd.concat(frames, ignore_index=True)
    return espn[(espn["season_type"] == REGULAR_SEASON) & espn["completed"]]


def match_games(games: pd.DataFrame, espn: pd.DataFrame) -> pd.DataFrame:
    """Join on date + the two teams in either order (neutral-site home labels may differ)."""
    pair = lambda frame: [frozenset(p) for p in zip(frame["home_team"], frame["away_team"])]
    games = games.assign(pair=pair(games))
    espn = espn.assign(pair=pair(espn))
    merged = games.merge(espn, left_on=["game_date", "pair"], right_on=["espn_date", "pair"], how="left",
                         suffixes=("", "_espn"))
    same_order = merged["home_team"] == merged["home_team_espn"]
    merged["espn_home_points"] = np.where(same_order, merged["home_score_espn"], merged["away_score_espn"])
    merged["espn_away_points"] = np.where(same_order, merged["away_score_espn"], merged["home_score_espn"])
    return merged


def feature_check(all_games: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Rebuild features with ESPN scores for 2025-26 and compare with the original features."""
    espn_scores = matched.set_index("game_id")[["espn_home_points", "espn_away_points"]].dropna()
    swapped = all_games.copy()
    rows = swapped["game_id"].isin(espn_scores.index)
    swapped.loc[rows, "home_score"] = swapped.loc[rows, "game_id"].map(espn_scores["espn_home_points"]).astype(int)
    swapped.loc[rows, "away_score"] = swapped.loc[rows, "game_id"].map(espn_scores["espn_away_points"]).astype(int)
    swapped["home_team_win"] = (swapped["home_score"] > swapped["away_score"]).astype(int)
    original = build_features_table(all_games).set_index("game_id")
    rebuilt = build_features_table(swapped).set_index("game_id")
    ids = original.index[original["season"] == SEASON]
    diff = (original.loc[ids, TEAM_FEATURES] - rebuilt.loc[ids, TEAM_FEATURES]).abs()
    return pd.DataFrame({"max_abs_difference": diff.max(), "games_differing": (diff > 1e-9).sum()})


def player_check(matched: pd.DataFrame) -> dict:
    sample = matched.dropna(subset=["espn_id"]).sample(PLAYER_SAMPLE_GAMES, random_state=SEED)
    logs = pd.read_csv(RAW_DATA_DIR / f"player_game_logs_{SEASON}.csv", dtype={"GAME_ID": str})
    logs["key"] = logs["PLAYER_NAME"].map(normalize_player_name)
    logs["game_score"] = game_score(logs)
    rows = []
    for game in sample.itertuples():
        espn = box_score(game.espn_id)
        espn["key"] = espn["player"].map(normalize_player_name)
        espn["game_score"] = game_score(espn)
        nba = logs[logs["GAME_ID"] == game.game_id]
        both = nba.merge(espn, on="key", how="outer", suffixes=("_nba", "_espn"), indicator=True)
        rows.append(both.assign(game_id=game.game_id))
    players = pd.concat(rows, ignore_index=True)
    matched_players = players[players["_merge"] == "both"]
    return {
        "games": PLAYER_SAMPLE_GAMES,
        "player rows (NBA logs)": int((players["_merge"] != "right_only").sum()),
        "matched by name": int(len(matched_players)),
        "only in NBA logs": players.loc[players["_merge"] == "left_only", "PLAYER_NAME"].dropna().unique().tolist()[:15],
        "only in ESPN": players.loc[players["_merge"] == "right_only", "player"].dropna().unique().tolist()[:15],
        "minutes identical": float((matched_players["MIN_nba"] == matched_players["MIN_espn"]).mean()),
        "minutes off by 1": float(((matched_players["MIN_nba"] - matched_players["MIN_espn"]).abs() == 1).mean()),
        "max minutes difference": int((matched_players["MIN_nba"] - matched_players["MIN_espn"]).abs().max()),
        "points identical": float((matched_players["PTS_nba"] == matched_players["PTS_espn"]).mean()),
        "game score identical (to 0.01)": float(((matched_players["game_score_nba"] - matched_players["game_score_espn"]).abs() < 0.01).mean()),
        "team identical": float((matched_players["TEAM_ABBREVIATION"] == matched_players["team"]).mean()),
    }


if __name__ == "__main__":
    all_games = load_games_table()
    games = all_games[all_games["season"] == SEASON].copy()
    espn = espn_season_games(games["game_date"])
    matched = match_games(games, espn)
    found = matched["espn_id"].notna()
    print(f"GAMES: {found.sum()} of {len(games)} matched to an ESPN game on the same date")
    print(matched.loc[~found, ["game_id", "game_date", "away_team", "home_team"]].to_string(index=False) if (~found).any() else "  (all matched)")
    m = matched[found]
    same_scores = (m["home_score"] == m["espn_home_points"]) & (m["away_score"] == m["espn_away_points"])
    print(f"SCORES identical: {same_scores.sum()} of {len(m)}")
    if not same_scores.all():
        print(m.loc[~same_scores, ["game_id", "home_team", "away_team", "home_score", "espn_home_points", "away_score", "espn_away_points"]].to_string(index=False))
    ours, theirs = m["is_neutral_site"].astype(bool), m["neutral_site"].astype(bool)
    print(f"NEUTRAL SITE: ours {ours.sum()}, ESPN {theirs.sum()}, disagreements {(ours != theirs).sum()}")
    print(m.loc[ours != theirs, ["game_id", "game_date", "away_team", "home_team", "is_neutral_site", "neutral_site"]].to_string(index=False))
    manifest = pd.read_csv(TEST_MANIFEST_PATH, dtype={"game_id": str}, parse_dates=["tip_off"])
    tips = m.merge(manifest[["game_id", "tip_off"]], on="game_id")
    official = tips["tip_off"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    gap = (tips["tip_off_utc"] - official).dt.total_seconds() / 60
    print(f"TIP-OFF vs official injury report: identical {(gap == 0).mean():.1%}, "
          f"within 30 min {(gap.abs() <= 30).mean():.1%}, max gap {gap.abs().max():.0f} min "
          f"(on {gap.notna().sum()} games with both)")
    print("\nFEATURES rebuilt from ESPN scores vs. original (2025-26):")
    print(feature_check(all_games, matched).to_string())
    print("\nPLAYERS (random sample of games):")
    for key, value in player_check(matched).items():
        print(f"  {key:32s} {value}")
