"""The daily live job: predict a day's games and publish them for the website.

    python -m src.live.predict                    predict today's games (Eastern date)
    python -m src.live.predict --day 2026-10-20   (re)build one day, e.g. a day the job missed

For each game, the injury report used is the newest one published at or before the CUTOFF:
    cutoff = tip-off - 60 minutes        (the earlier of ESPN's and the injury report's tip-off time)
Before the cutoff the job makes a PREVIEW from the latest report so far; it is replaced on later runs.
At or after the cutoff the prediction is FINAL: it is locked and never changed again, and only final
predictions count in the track record. Team features use only games finished on earlier dates, so a day
can be rebuilt later with exactly the information that existed before its games.

Outputs:
    live_state/predictions_<season>.csv    every prediction (the track record), committed by the job
    web/public/data/predictions.json       what the website reads (every game, summary)
    web/public/data/games/<game_id>.json   one game's details page: Elo, form, rest, injured players
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.collect_data import PROJECT_ROOT
from src.live.collect import (EASTERN, latest_report, listed_for_games, load_listings, parsed, season_listings,
                              season_logs, update_box_scores, update_listings)
from src.live.espn import scoreboard
from src.live.injury_features import (INJURY_INPUTS, LISTING_STATE_PATH, VALUE_STATE_PATH, injury_diffs,
                                      injury_inputs)
from src.live.model import load_model, predict
from src.live.team_features import (STATE_DIR, TEAM_INPUTS, UNPLAYED_STATUSES, espn_season_games, load_history,
                                    team_features)

SEASON = "2026-27"
SEASON_START = pd.Timestamp("2026-10-20")
MINUTES_BEFORE_TIP = 60
PREDICTIONS_PATH = STATE_DIR / f"predictions_{SEASON}.csv"
SITE_DATA_PATH = PROJECT_ROOT / "web" / "public" / "data" / "predictions.json"
GAME_DETAILS_DIR = SITE_DATA_PATH.parent / "games"
INPUTS = TEAM_INPUTS + [f"{name}_diff" for name in INJURY_INPUTS]
PREDICTION_COLUMNS = ["game_id", "game_date", "tip_off_et", "home_team", "away_team", "p_home", "final",
                      "report", "predicted_at", *INPUTS]


def to_eastern(times) -> pd.Series:
    return pd.to_datetime(times, utc=True).dt.tz_convert(EASTERN).dt.tz_localize(None)


def unfinished_games(day: pd.Timestamp) -> int:
    """Games on `day` that started but have not finished (their results are still unknown)."""
    board = scoreboard(day, cache=False)
    if board.empty:
        return 0
    return int((~board["completed"] & ~board["status"].isin(UNPLAYED_STATUSES)
                & (board["status"] != "Scheduled")).sum())


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.read_csv(PREDICTIONS_PATH, dtype={"game_id": str}, parse_dates=["game_date", "tip_off_et"])


def cutoffs(games: pd.DataFrame, report: pd.DataFrame) -> pd.Series:
    """Tip-off minus 60 minutes, using the earlier of ESPN's time and the injury report's time."""
    tips = to_eastern(games["tip_off_utc"]).to_numpy()
    if len(report):
        listed = report.drop_duplicates("matchup").set_index("matchup")["tip_off"]
        report_tips = (games["away_team"] + "@" + games["home_team"]).map(listed)
        same_day = report_tips.dt.normalize() == games["game_date"]
        tips = np.where(same_day & (report_tips < tips), report_tips, tips)
    return pd.Series(pd.to_datetime(tips), index=games.index) - pd.Timedelta(minutes=MINUTES_BEFORE_TIP)


def injury_table(day, games, now_et, box_scores):
    """Injury inputs (away minus home) per game; the report, final flag and cutoff used for each game;
    and, for the website, each listed player's numbers and which teams' reports were known."""
    # Tip-off times come from the report published before ESPN's cutoff (it always lists the game), not from
    # the newest report: that way a game's cutoff is the same whenever the job runs, even days later.
    espn_cutoff = to_eastern(games["tip_off_utc"]) - pd.Timedelta(minutes=MINUTES_BEFORE_TIP)
    probe = espn_cutoff.where(espn_cutoff < now_et, now_et)
    games = games.assign(cutoff=espn_cutoff)
    for as_of in probe.unique():
        stamp = latest_report(as_of)
        if stamp:
            same = probe == as_of
            games.loc[same, "cutoff"] = cutoffs(games[same], parsed(stamp))
    games["final"] = now_et >= games["cutoff"]
    games["as_of"] = games["cutoff"].where(games["final"], now_et)

    stamps = {as_of: latest_report(as_of) for as_of in games["as_of"].unique()}
    games["report"] = games["as_of"].map(stamps)
    listed, known = [], []
    for (stamp, as_of), group in games.groupby(["report", "as_of"], dropna=False):
        if pd.isna(stamp):
            continue
        group_listed, group_known = listed_for_games(group, parsed(stamp), as_of)
        update_listings(SEASON, group_listed, group, stamp, final=bool(group["final"].iloc[0]))
        listed.append(group_listed)
        known.append(group_known)
    listed = pd.concat(listed, ignore_index=True) if listed else pd.DataFrame(
        columns=["game_id", "team_abbr", "player", "status"])
    known = pd.concat(known, ignore_index=True) if known else pd.DataFrame(columns=["game_id", "team_abbr", "known"])

    past = box_scores[box_scores["game_date"] < day]
    listings = season_listings(load_listings(SEASON), box_scores)
    per_team, players = injury_inputs(day, SEASON, listed, season_logs(past, SEASON),
                                      listings[listings["game_date"] < day].drop(columns="game_id"),
                                      pd.read_csv(VALUE_STATE_PATH), pd.read_csv(LISTING_STATE_PATH),
                                      with_players=True)
    diffs = injury_diffs(games, per_team, known)
    return (diffs[["game_id", *[f"{name}_diff" for name in INJURY_INPUTS]]],
            games[["game_id", "final", "report", "cutoff"]], players, known)


def run(day: pd.Timestamp | None = None, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now_et = to_eastern(pd.Series([now or pd.Timestamp.now(tz="UTC")])).iloc[0]
    today = now_et.normalize()
    if day is None:
        # After midnight, while last night's games are still being played, keep working on last night
        # (its games are locked; this run only adds results) instead of starting the new day too early.
        yesterday = today - pd.Timedelta(days=1)
        day = yesterday if yesterday >= SEASON_START and unfinished_games(yesterday) else today
    day = pd.Timestamp(day)
    if day > today:
        raise ValueError("only today or past days can be predicted")
    if day > SEASON_START and unfinished_games(day - pd.Timedelta(days=1)):
        raise RuntimeError(f"games on {day - pd.Timedelta(days=1):%Y-%m-%d} are still being played; run again later")

    season_games = espn_season_games(SEASON, SEASON_START, today, today)
    box_scores = update_box_scores(SEASON, season_games[season_games["home_score"].notna()][["game_id", "game_date"]])

    saved = load_predictions()
    locked = set(saved.loc[saved["final"].astype(bool), "game_id"])
    # The day's games are predicted from earlier days only: hide the day's results (if any) from the features.
    known_games = season_games[season_games["game_date"] < day]
    day_games = season_games[season_games["game_date"] == day].assign(home_score=np.nan, away_score=np.nan,
                                                                      home_team_win=np.nan)
    todo = day_games[~day_games["game_id"].isin(locked)]
    # Locked games without a details page (e.g. locked before details existed): recompute them only to write
    # the page. Their locked prediction is never changed.
    backfill = day_games[day_games["game_id"].isin(locked)
                         & ~day_games["game_id"].map(lambda g: (GAME_DETAILS_DIR / f"{g}.json").exists())]
    compute = pd.concat([todo, backfill], ignore_index=True)
    rows = pd.DataFrame(columns=PREDICTION_COLUMNS)
    if not compute.empty:
        team = team_features(load_history(), pd.concat([known_games, compute], ignore_index=True), details=True)
        injury, flags, players, known = injury_table(day, compute, now_et, box_scores)
        rows = compute.merge(team, on="game_id").merge(injury, on="game_id").merge(flags, on="game_id")
        rows = rows.assign(p_home=predict(load_model(), rows), tip_off_et=to_eastern(rows["tip_off_utc"]),
                           predicted_at=now_et.isoformat(timespec="minutes"))
        refilled = rows[rows["game_id"].isin(backfill["game_id"])]
        write_game_details(matching_locked(refilled, saved), players, known)
        rows = rows[rows["game_id"].isin(todo["game_id"])]
        rows = rows[np.array([changed(row, saved) for _, row in rows.iterrows()], dtype=bool)]
        write_game_details(rows, players, known)
        rows = rows[PREDICTION_COLUMNS]
    if not rows.empty:
        saved = pd.concat([saved[~saved["game_id"].isin(rows["game_id"])], rows], ignore_index=True)
        saved = saved.sort_values(["game_date", "tip_off_et", "game_id"]).reset_index(drop=True)
        PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
        saved.to_csv(PREDICTIONS_PATH, index=False)

    write_site_data(saved, season_games, now_et)
    return saved[saved["game_date"] == day]


def matching_locked(rows: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    """Recomputed locked games whose probability and report equal the locked ones (the details then describe
    exactly the published prediction). Any mismatch is reported and skipped, never written over."""
    locked = saved.set_index("game_id")
    keep = []
    for _, row in rows.iterrows():
        old = locked.loc[row["game_id"]]
        same = np.isclose(float(old["p_home"]), row["p_home"], atol=1e-9) and old["report"] == row["report"]
        if not same:
            print(f"warning: {row['game_id']} recomputes to {row['p_home']:.4f} (locked {float(old['p_home']):.4f}); "
                  "details page skipped")
        keep.append(same)
    # The page shows when the locked prediction was made, not when the page was backfilled.
    rows = rows[np.array(keep, dtype=bool)]
    return rows.assign(predicted_at=[locked.loc[g, "predicted_at"] for g in rows["game_id"]],
                             final=True)


def changed(row: pd.Series, saved: pd.DataFrame) -> bool:
    """New game, or a preview whose probability or report changed (unchanged rows are not rewritten,
    so the job only commits when something real changed)."""
    before = saved[saved["game_id"] == row["game_id"]]
    if before.empty:
        return True
    old = before.iloc[0]
    return (not np.isclose(float(old["p_home"]), row["p_home"], atol=1e-9) or old["report"] != row["report"]
            or bool(old["final"]) != bool(row["final"]))


# ---------------------------------------------------------------- website data

def number(value, digits: int = 3):
    return None if pd.isna(value) else round(float(value), digits)


def write_game_details(rows: pd.DataFrame, players: pd.DataFrame, known: pd.DataFrame) -> None:
    """One JSON file per game with the raw numbers behind its prediction (NaN = no games yet this season)."""
    GAME_DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    is_known = known.set_index(["game_id", "team_abbr"])["known"] if len(known) else pd.Series(dtype=bool)
    for _, game in rows.iterrows():
        sides = {}
        for side in ["home", "away"]:
            team = game[f"{side}_team"]
            listed = players[(players["game_id"] == game["game_id"]) & (players["team_abbr"] == team)]
            listed = listed.sort_values("exp_missing_minutes", ascending=False)
            sides[side] = {
                "team": team,
                "elo": number(game[f"{side}_elo"], 1),
                "win_pct_last10": number(game[f"{side}_win_pct_last10"]),
                "win_pct_last5": number(game[f"{side}_win_pct_last5"]),
                "point_diff_last10": number(game[f"{side}_point_diff_last10"], 1),
                "rest_days": number(game[f"{side}_rest_days"], 0),
                "back_to_back": bool(game[f"{side}_back_to_back"] == 1),
                "report_known": bool(is_known.get((game["game_id"], team), False)),
                "injuries": [{"player": p.player, "status": p.status, "avg_minutes": number(p.avg_minutes, 1),
                              "p_play": number(p.p_play), "exp_missing_minutes": number(p.exp_missing_minutes, 1)}
                             for p in listed.itertuples()],
            }
        details = {"id": game["game_id"], "computed_et": game["predicted_at"], "final": bool(game["final"]),
                   "report": None if pd.isna(game["report"]) else game["report"],
                   "cutoff_et": None if pd.isna(game["cutoff"]) else game["cutoff"].isoformat(timespec="minutes"),
                   "elo_home_win_prob": number(game["elo_home_win_prob"]),
                   "inputs": {name: number(game[name], 4) for name in INPUTS}, **sides}
        (GAME_DETAILS_DIR / f"{game['game_id']}.json").write_text(json.dumps(details, indent=1) + "\n")


def contributions(row: pd.Series, model: dict) -> dict:
    """Each input's push on the home team's log-odds (positive helps the home team)."""
    return {name: round(model["weight"][name] * float(row[name]) / model["scale"][name], 4) for name in model["inputs"]}


def track_record(games: pd.DataFrame) -> dict:
    scored = games[games["final"] & games["home_score"].notna()]
    if scored.empty:
        return {"games": 0}
    p, won = scored["p_home"].to_numpy(dtype=float), (scored["home_score"] > scored["away_score"]).to_numpy()
    eps = 1e-12
    return {"games": int(len(scored)), "correct": int(((p >= 0.5) == won).sum()),
            "accuracy": round(float(((p >= 0.5) == won).mean()), 4),
            "log_loss": round(float(-np.mean(np.where(won, np.log(p + eps), np.log(1 - p + eps)))), 4),
            "brier": round(float(np.mean((p - won) ** 2)), 4)}


def write_site_data(saved: pd.DataFrame, season_games: pd.DataFrame, now_et: pd.Timestamp) -> None:
    model = load_model()
    results = season_games[season_games["home_score"].notna()][["game_id", "home_score", "away_score"]]
    games = saved.merge(results, on="game_id", how="left")
    games["final"] = games["final"].astype(bool)
    rows = [{"id": g.game_id, "date": f"{g.game_date:%Y-%m-%d}", "tip_off_et": g.tip_off_et.isoformat(timespec="minutes"),
             "home": g.home_team, "away": g.away_team, "p_home": round(float(g.p_home), 4), "final": bool(g.final),
             "report": None if pd.isna(g.report) else g.report,
             "home_score": None if pd.isna(g.home_score) else int(g.home_score),
             "away_score": None if pd.isna(g.away_score) else int(g.away_score),
             "contributions": contributions(g, model)} for _, g in games.iterrows()]
    data = {"updated_et": now_et.isoformat(timespec="minutes"), "season": SEASON,
            "model": {"description": model["description"], "training_games": model["training_games"],
                      "trained_on": model["trained_on"]},
            "record": track_record(games), "games": rows}
    if SITE_DATA_PATH.exists():
        old = json.loads(SITE_DATA_PATH.read_text())
        if {**old, "updated_et": None} == {**data, "updated_et": None}:
            return                            # nothing new: keep the file (and skip a commit)
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA_PATH.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", help="YYYY-MM-DD (default: today, Eastern time)")
    args = parser.parse_args()
    predictions = run(args.day)
    print(f"{len(predictions)} games:")
    for g in predictions.itertuples():
        kind = "FINAL  " if g.final else "preview"
        print(f"  {kind} {g.away_team} @ {g.home_team}  P(home wins) = {g.p_home:.1%}  report {g.report}")
