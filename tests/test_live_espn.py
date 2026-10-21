"""Tests for the ESPN reader (src/live/espn.py) on hand-made responses; no network."""

import pandas as pd

import src.live.espn as espn


def fake_get_json(payload):
    return lambda path, params, cache_name=None: payload


def test_scoreboard_maps_team_codes_and_reads_scores(monkeypatch):
    payload = {"events": [{
        "id": "1", "date": "2026-10-21T01:30Z", "season": {"type": 2},
        "status": {"type": {"completed": True, "description": "Final"}},
        "competitions": [{"neutralSite": False, "competitors": [
            {"homeAway": "home", "team": {"abbreviation": "SA"}, "score": "111"},
            {"homeAway": "away", "team": {"abbreviation": "OKC"}, "score": "109"}]}],
    }]}
    monkeypatch.setattr(espn, "get_json", fake_get_json(payload))
    game = espn.scoreboard(pd.Timestamp("2026-10-20")).iloc[0]
    assert (game.home_team, game.away_team) == ("SAS", "OKC")
    assert (game.home_score, game.away_score, game.completed) == (111, 109, True)
    assert game.tip_off_utc == pd.Timestamp("2026-10-21T01:30Z")


def test_box_score_parses_made_attempted_and_skips_players_without_minutes(monkeypatch):
    labels = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
    payload = {"boxscore": {"players": [{"team": {"abbreviation": "GS"}, "statistics": [{
        "labels": labels, "athletes": [
            {"athlete": {"displayName": "Played Player"},
             "stats": ["31", "38", "12-21", "7-12", "7-8", "5", "3", "3", "5", "0", "0", "5", "5", "+9"]},
            {"athlete": {"displayName": "Bench Player"}, "stats": ["--"] * 14},
            {"athlete": {"displayName": "Injured Player"}, "stats": [], "didNotPlay": True},
        ]}]}]}}
    monkeypatch.setattr(espn, "get_json", fake_get_json(payload))
    box = espn.box_score("1")
    assert list(box["player"]) == ["Played Player"]
    row = box.iloc[0]
    assert (row.team, row.MIN, row.PTS, row.FGM, row.FGA, row.FG3M, row.FTA, row.TOV, row.PLUS_MINUS) == \
           ("GSW", 31, 38, 12, 21, 7, 8, 3, 9)
