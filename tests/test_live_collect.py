"""Offline tests for live data collection (no internet: the report site is faked)."""

import pandas as pd
import pytest

import src.live.collect as collect


class FakeSite:
    """Answers HEAD requests: 200 for the given stamps, 403 otherwise."""
    def __init__(self, stamps):
        self.stamps, self.asked = set(stamps), []

    def head(self, url, timeout):
        stamp = url.split("Injury-Report_")[1].removesuffix(".pdf")
        self.asked.append(stamp)
        return type("Response", (), {"status_code": 200 if stamp in self.stamps else 403})()


def test_report_stamp_uses_quarter_hour_format():
    assert collect.report_stamp(pd.Timestamp("2026-10-20 17:00")) == "2026-10-20_05_00PM"
    assert collect.report_stamp(pd.Timestamp("2026-10-21 09:45")) == "2026-10-21_09_45AM"


def test_latest_report_walks_back_to_newest_report_at_or_before_cutoff():
    site = FakeSite(["2026-10-20_05_00PM", "2026-10-20_06_30PM", "2026-10-20_07_00PM"])
    assert collect.latest_report(pd.Timestamp("2026-10-20 18:59"), site) == "2026-10-20_06_30PM"
    assert "2026-10-20_07_00PM" not in site.asked          # never looks past the cutoff


def test_latest_report_falls_back_to_previous_evening_and_skips_night_slots():
    site = FakeSite(["2026-10-20_09_45PM"])
    assert collect.latest_report(pd.Timestamp("2026-10-21 10:10"), site) == "2026-10-20_09_45PM"
    assert not any(s.startswith("2026-10-21") and s.endswith("AM") and s < "2026-10-21_09" for s in site.asked)


def report_rows(printed="2026-10-20 18:00"):
    rows = [("BOS@DET", "Boston Celtics", "Carter, Devin", "Out"),
            ("BOS@DET", "Detroit Pistons", "Jones, Isaac", "Questionable"),
            ("PHI@NYK", "Philadelphia 76ers", None, "Not yet submitted")]
    return pd.DataFrame([{"matchup": m, "team": t, "player": p, "status": s, "report_time": pd.Timestamp(printed),
                          "tip_off": pd.Timestamp("2026-10-20 19:00")} for m, t, p, s in rows])


GAMES = pd.DataFrame({"game_id": ["espn-1", "espn-2", "espn-3"], "game_date": pd.Timestamp("2026-10-20"),
                      "home_team": ["DET", "NYK", "SAS"], "away_team": ["BOS", "PHI", "OKC"]})


def test_listed_for_games_players_and_known_flags():
    listed, known = collect.listed_for_games(GAMES, report_rows(), pd.Timestamp("2026-10-20 18:00"))
    assert listed["player"].tolist() == ["Carter, Devin", "Jones, Isaac"]
    flags = known.set_index(["game_id", "team_abbr"])["known"]
    assert flags[("espn-1", "DET")] and flags[("espn-1", "BOS")]
    assert flags[("espn-2", "NYK")] and not flags[("espn-2", "PHI")]      # not yet submitted
    assert not flags[("espn-3", "SAS")] and not flags[("espn-3", "OKC")]  # game missing from report


def test_listed_for_games_rejects_report_printed_after_cutoff():
    with pytest.raises(ValueError, match="after the cutoff"):
        collect.listed_for_games(GAMES, report_rows("2026-10-20 18:15"), pd.Timestamp("2026-10-20 18:00"))


def test_update_listings_keeps_final_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "STATE_DIR", tmp_path)
    games = GAMES.iloc[:1]
    first = pd.DataFrame({"game_id": ["espn-1"], "team_abbr": ["BOS"], "player": ["Carter, Devin"], "status": ["Out"]})
    collect.update_listings("2026-27", first.assign(status="Questionable"), games, "morning", final=False)
    collect.update_listings("2026-27", first, games, "pre-tip", final=True)          # replaces the morning one
    saved = collect.update_listings("2026-27", first.iloc[:0], games, "late", final=False)  # final one is kept
    assert saved[["status", "report"]].values.tolist() == [["Out", "pre-tip"]]


def test_season_listings_played_flag_ignores_name_suffixes():
    listings = pd.DataFrame({"game_id": ["espn-1", "espn-1"], "game_date": pd.Timestamp("2026-10-20"),
                             "player": ["Portis Jr., Bobby", "Carter, Devin"], "status": ["Probable", "Out"]})
    box = pd.DataFrame({"game_id": ["espn-1"], "player": ["Bobby Portis"]})
    assert collect.season_listings(listings, box)["played"].tolist() == [True, False]
