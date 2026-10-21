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
    web/public/data/predictions.json       what the website reads
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


def injury_table(day, games, now_et, box_scores) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injury inputs (away minus home) per game, and the report / final flag used for each game."""
    newest = latest_report(now_et)
    report_now = parsed(newest) if newest else pd.DataFrame(columns=["matchup", "tip_off"])
    games = games.assign(cutoff=cutoffs(games, report_now))
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
    per_team = injury_inputs(day, SEASON, listed, season_logs(past, SEASON),
                             listings[listings["game_date"] < day].drop(columns="game_id"),
                             pd.read_csv(VALUE_STATE_PATH), pd.read_csv(LISTING_STATE_PATH))
    diffs = injury_diffs(games, per_team, known)
    return diffs[["game_id", *[f"{name}_diff" for name in INJURY_INPUTS]]], games[["game_id", "final", "report"]]


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
    if not todo.empty:
        team = team_features(load_history(), pd.concat([known_games, todo], ignore_index=True))
        injury, flags = injury_table(day, todo.reset_index(drop=True), now_et, box_scores)
        rows = todo.merge(team, on="game_id").merge(injury, on="game_id").merge(flags, on="game_id")
        rows = rows.assign(p_home=predict(load_model(), rows), tip_off_et=to_eastern(rows["tip_off_utc"]),
                           predicted_at=now_et.isoformat(timespec="minutes"))[PREDICTION_COLUMNS]
        rows = rows[[changed(row, saved) for _, row in rows.iterrows()]]
    if not todo.empty and not rows.empty:
        saved = pd.concat([saved[~saved["game_id"].isin(rows["game_id"])], rows], ignore_index=True)
        saved = saved.sort_values(["game_date", "tip_off_et", "game_id"]).reset_index(drop=True)
        PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
        saved.to_csv(PREDICTIONS_PATH, index=False)

    write_site_data(saved, season_games, now_et)
    return saved[saved["game_date"] == day]


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
