"""Offline tests for the live predictor's bookkeeping (cutoffs, locking, track record)."""

import pandas as pd

from src.live.predict import changed, cutoffs, track_record


def test_cutoff_uses_earlier_of_espn_and_report_tip_off():
    games = pd.DataFrame({"home_team": ["SAS", "NYK"], "away_team": ["OKC", "PHI"],
                          "game_date": pd.Timestamp("2026-10-20"),
                          "tip_off_utc": ["2026-10-21T01:50Z", "2026-10-20T23:00Z"]})   # 9:50 PM and 7:00 PM ET
    report = pd.DataFrame({"matchup": ["OKC@SAS", "PHI@NYK"],
                           "tip_off": pd.to_datetime(["2026-10-20 21:30", "2026-10-20 19:30"])})
    assert cutoffs(games, report).tolist() == [pd.Timestamp("2026-10-20 20:30"), pd.Timestamp("2026-10-20 18:00")]


def test_changed_only_when_probability_report_or_lock_changes():
    saved = pd.DataFrame({"game_id": ["espn-1"], "p_home": [0.6], "report": ["r1"], "final": [False]})
    row = lambda **kw: pd.Series({"game_id": "espn-1", "p_home": 0.6, "report": "r1", "final": False, **kw})
    assert not changed(row(), saved)
    assert changed(row(p_home=0.61), saved)
    assert changed(row(report="r2"), saved)
    assert changed(row(final=True), saved)
    assert changed(row(game_id="espn-2"), saved)


def test_track_record_counts_only_locked_games_with_results():
    games = pd.DataFrame({"final": [True, True, False, True], "p_home": [0.7, 0.4, 0.9, 0.6],
                          "home_score": [110, 100, 90, None], "away_score": [100, 105, 120, None]})
    record = track_record(games)
    assert record["games"] == 2 and record["correct"] == 2 and record["accuracy"] == 1.0


def test_backfilled_details_only_for_games_that_reproduce_the_locked_prediction(capsys):
    from src.live.predict import matching_locked
    saved = pd.DataFrame({"game_id": ["espn-1", "espn-2"], "p_home": [0.6, 0.7], "report": ["r1", "r1"],
                          "final": [True, True], "predicted_at": ["2026-10-20T18:00", "2026-10-20T18:00"]})
    recomputed = pd.DataFrame({"game_id": ["espn-1", "espn-2"], "p_home": [0.6, 0.71], "report": ["r1", "r1"],
                               "final": [True, True], "predicted_at": ["2026-10-22T10:00"] * 2})
    kept = matching_locked(recomputed, saved)
    assert kept["game_id"].tolist() == ["espn-1"]
    assert kept["predicted_at"].tolist() == ["2026-10-20T18:00"]     # when it was locked, not when backfilled
    assert "espn-2" in capsys.readouterr().out                         # mismatch is reported, not written
