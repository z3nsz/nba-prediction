"""Read NBA schedules, scores and player box scores from ESPN's public site API.

Used by the live prediction system because stats.nba.com does not answer requests
from cloud servers (e.g. GitHub Actions), while ESPN does.

    scoreboard: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
    game:       https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>

ESPN uses a few different team codes; they are mapped to the NBA codes our data uses.
Responses for finished games are cached in data/raw/espn/ (they never change).
"""

import json
import time

import pandas as pd
import requests

from src.collect_data import RAW_DATA_DIR

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE_DIR = RAW_DATA_DIR / "espn"
ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
MAX_ATTEMPTS = 4
PAUSE_SECONDS = 0.2
REGULAR_SEASON = 2  # ESPN season type: 1 preseason, 2 regular season, 3 postseason


def nba_code(espn_code: str) -> str:
    return ESPN_TO_NBA.get(espn_code, espn_code)


def get_json(path: str, params: dict, cache_name: str | None = None) -> dict:
    """GET an ESPN endpoint with retries; cache to `cache_name` if given (finished games only)."""
    cache = CACHE_DIR / cache_name if cache_name else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                time.sleep(PAUSE_SECONDS)
                if cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(data))
                return data
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"ESPN request failed: {path} {params}")


def scoreboard(date: pd.Timestamp, cache: bool = True) -> pd.DataFrame:
    """Every game ESPN lists for one date: id, tip-off (UTC), teams, scores, status, neutral site."""
    data = get_json("/scoreboard", {"dates": f"{date:%Y%m%d}"},
                    f"scoreboard_{date:%Y%m%d}.json" if cache else None)
    rows = []
    for event in data.get("events", []):
        competition = event["competitions"][0]
        teams = {team["homeAway"]: team for team in competition["competitors"]}
        rows.append({
            "espn_id": event["id"],
            "tip_off_utc": pd.Timestamp(event["date"]),
            "season_type": event.get("season", {}).get("type"),
            "home_team": nba_code(teams["home"]["team"]["abbreviation"]),
            "away_team": nba_code(teams["away"]["team"]["abbreviation"]),
            "home_score": int(teams["home"].get("score") or 0),
            "away_score": int(teams["away"].get("score") or 0),
            "completed": bool(event["status"]["type"].get("completed")),
            "status": event["status"]["type"].get("description"),
            "neutral_site": bool(competition.get("neutralSite")),
            "note": " | ".join(n.get("headline", "") for n in competition.get("notes", [])),
        })
    return pd.DataFrame(rows)


def _made_attempted(text: str) -> tuple[int, int]:
    made, attempted = text.split("-")
    return int(made), int(attempted)


def box_score(espn_id: str, cache: bool = True) -> pd.DataFrame:
    """Player box score for one game, with the columns our player game logs use."""
    data = get_json("/summary", {"event": espn_id}, f"summary_{espn_id}.json" if cache else None)
    rows = []
    for team in data.get("boxscore", {}).get("players", []):
        code = nba_code(team["team"]["abbreviation"])
        block = team["statistics"][0]
        labels = block["labels"]
        for athlete in block["athletes"]:
            stats = dict(zip(labels, athlete.get("stats") or []))
            # Players listed without minutes ("--") did not play; our NBA player logs leave them out too.
            if not stats or athlete.get("didNotPlay") or not str(stats.get("MIN", "")).isdigit():
                continue
            fgm, fga = _made_attempted(stats["FG"])
            fg3m, fg3a = _made_attempted(stats["3PT"])
            ftm, fta = _made_attempted(stats["FT"])
            rows.append({
                "espn_id": espn_id, "team": code, "player": athlete["athlete"]["displayName"],
                "MIN": int(stats["MIN"] or 0), "PTS": int(stats["PTS"]), "FGM": fgm, "FGA": fga,
                "FG3M": fg3m, "FG3A": fg3a, "FTM": ftm, "FTA": fta,
                "OREB": int(stats["OREB"]), "DREB": int(stats["DREB"]), "REB": int(stats["REB"]),
                "AST": int(stats["AST"]), "STL": int(stats["STL"]), "BLK": int(stats["BLK"]),
                "TOV": int(stats["TO"]), "PF": int(stats["PF"]),
                "PLUS_MINUS": int(str(stats.get("+/-", "0")).replace("+", "") or 0),
            })
    return pd.DataFrame(rows)
