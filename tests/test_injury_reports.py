"""Tests for the injury report parser (src/injury_reports.py).

Parser tests run on the sample PDFs in data/raw/injury_reports/ and are skipped
if they have not been downloaded (the PDFs are not committed).
"""

import pandas as pd
import pytest

from src.injury_reports import INJURY_REPORT_DIR, STATUSES, normalize_player_name, parse_report

TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


@pytest.fixture(scope="module")
def parsed_samples():
    paths = sorted(INJURY_REPORT_DIR.glob("*.pdf"))
    if not paths:
        pytest.skip("no sample injury reports downloaded")
    # Check an evenly spread sample (always the same files), not all ~2,400 reports.
    step = max(1, len(paths) // 20)
    paths = paths[::step][:20]
    return pd.concat([parse_report(path) for path in paths], ignore_index=True)


def test_every_row_belongs_to_a_team_in_its_matchup(parsed_samples):
    """Catches carry-forward bugs, like a 'Page 2 of 9' footer becoming a team name."""
    in_matchup = parsed_samples.apply(
        lambda row: row["team"] in {TEAM_NAMES[abbr] for abbr in row["matchup"].split("@")}, axis=1)
    assert in_matchup.all()


def test_statuses_times_and_names_are_parsed(parsed_samples):
    assert parsed_samples["status"].isin(STATUSES + ["Not yet submitted"]).all()
    assert parsed_samples["report_time"].notna().all()
    assert parsed_samples["tip_off"].notna().all()
    assert parsed_samples["tip_off"].dt.hour.between(11, 23).all(), "tip-offs are between 11 AM and 11 PM ET"
    players = parsed_samples["player"].dropna()
    assert players.str.contains(",").all(), "names are printed 'Last, First'"


@pytest.mark.parametrize("report_name, log_name", [
    ("Harper Jr., Ron", "Ron Harper Jr."),
    ("Antetokounmpo, Giannis", "Giannis Antetokounmpo"),
    ("Doncic, Luka", "Luka Dončić"),
    ("O'Neale, Royce", "Royce O'Neale"),
])
def test_name_formats_normalize_to_the_same_key(report_name, log_name):
    assert normalize_player_name(report_name) == normalize_player_name(log_name)


def test_tip_off_times_without_am_pm_are_all_pm():
    """'12:30' is 12:30 PM, '07:00' is 7 PM and '11:00' is 11 PM (late West Coast games), never 11 AM."""
    from src.injury_reports import tip_off_times
    times = tip_off_times(pd.Series(["03/24/2026"] * 3), pd.Series(["12:30 (ET)", "07:00 (ET)", "11:00 (ET)"]))
    assert [t.strftime("%H:%M") for t in times] == ["12:30", "19:00", "23:00"]
