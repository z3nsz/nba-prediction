"""Pre-game travel and fatigue features (Step 3.2b).

Every game is played at a venue: the home team's arena, except
    - neutral-site games (Mexico City, Paris, Las Vegas, Berlin, London)
    - the 2019-20 restart in the Orlando "bubble" (Elo warm-up season only)
    - Toronto's 2020-21 home games, played in Tampa because of COVID border rules

From each team's sequence of venues, before every game:
    travel_km           great-circle distance from the previous game's venue
    travel_km_last7d    total distance traveled for games in the last 7 days (incl. this trip)
    tz_change_hours     clock change since the previous game (+ = moved east, lost hours)
    road_trip_game      1st, 2nd, 3rd... game of the current road trip (0 = at home)

Schedules and venues are published months in advance, so none of this is
leakage. Coordinates are approximate arena locations (well within 10 km), and
time zones are looked up for each game's actual date, which handles daylight
saving time (Arizona does not observe it).

Venue facts are from general public knowledge, not an API; tests check that
every neutral-site game in the data has a venue assigned.
"""

from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# team -> (latitude, longitude, time zone) of its home arena
ARENAS = {
    "ATL": (33.757, -84.396, "America/New_York"), "BOS": (42.366, -71.062, "America/New_York"),
    "BKN": (40.683, -73.975, "America/New_York"), "CHA": (35.225, -80.839, "America/New_York"),
    "CHI": (41.881, -87.674, "America/Chicago"), "CLE": (41.496, -81.688, "America/New_York"),
    "DAL": (32.790, -96.810, "America/Chicago"), "DEN": (39.749, -105.008, "America/Denver"),
    "DET": (42.341, -83.055, "America/Detroit"), "GSW": (37.768, -122.388, "America/Los_Angeles"),
    "HOU": (29.751, -95.362, "America/Chicago"), "IND": (39.764, -86.155, "America/Indiana/Indianapolis"),
    "LAC": (34.043, -118.267, "America/Los_Angeles"), "LAL": (34.043, -118.267, "America/Los_Angeles"),
    "MEM": (35.138, -90.051, "America/Chicago"), "MIA": (25.781, -80.188, "America/New_York"),
    "MIL": (43.045, -87.917, "America/Chicago"), "MIN": (44.979, -93.276, "America/Chicago"),
    "NOP": (29.949, -90.082, "America/Chicago"), "NYK": (40.751, -73.993, "America/New_York"),
    "OKC": (35.463, -97.515, "America/Chicago"), "ORL": (28.539, -81.384, "America/New_York"),
    "PHI": (39.901, -75.172, "America/New_York"), "PHX": (33.446, -112.071, "America/Phoenix"),
    "POR": (45.532, -122.667, "America/Los_Angeles"), "SAC": (38.580, -121.500, "America/Los_Angeles"),
    "SAS": (29.427, -98.438, "America/Chicago"), "TOR": (43.643, -79.379, "America/Toronto"),
    "UTA": (40.768, -111.901, "America/Denver"), "WAS": (38.898, -77.021, "America/New_York"),
}
OTHER_VENUES = {
    "Tampa": (27.943, -82.452, "America/New_York"),
    "Orlando bubble": (28.337, -81.546, "America/New_York"),
    "Mexico City": (19.404, -99.096, "America/Mexico_City"),
    "Paris": (48.839, 2.379, "Europe/Paris"),
    "Las Vegas": (36.103, -115.178, "America/Los_Angeles"),
    "Berlin": (52.508, 13.443, "Europe/Berlin"),
    "London": (51.503, 0.003, "Europe/London"),
}
NEUTRAL_SITE_VENUES = {
    "0022200439": "Mexico City", "0022200678": "Paris", "0022300172": "Mexico City",
    "0022301229": "Las Vegas", "0022301230": "Las Vegas", "0022300527": "Paris",
    "0022400147": "Mexico City", "0022401229": "Las Vegas", "0022401230": "Las Vegas",
    "0022400621": "Paris", "0022400633": "Paris",
    "0022500147": "Mexico City", "0022501229": "Las Vegas", "0022501230": "Las Vegas",
    "0022500578": "Berlin", "0022500602": "London",
}
BUBBLE_SEASON, BUBBLE_START = "2019-20", "2020-07-30"
TAMPA_SEASON = "2020-21"
EARTH_RADIUS_KM = 6371.0

TRAVEL_FEATURES = ["travel_km", "travel_km_last7d", "tz_change_hours", "road_trip_game"]


def game_venues(games: pd.DataFrame) -> pd.Series:
    """Venue name for every game (a team abbreviation means that team's arena)."""
    venue = games["home_team"].copy()
    venue[(games["season"] == TAMPA_SEASON) & (games["home_team"] == "TOR")] = "Tampa"
    venue[(games["season"] == BUBBLE_SEASON) & (games["game_date"] >= BUBBLE_START)] = "Orlando bubble"
    mapped = games["game_id"].map(NEUTRAL_SITE_VENUES)
    venue = mapped.fillna(venue)

    unmapped = games["is_neutral_site"].eq(1) & ~venue.isin(OTHER_VENUES)
    if unmapped.any():
        raise ValueError(f"neutral-site games without a venue: {games.loc[unmapped, 'game_id'].tolist()}")
    return venue


def venue_info(venue: str) -> tuple[float, float, str]:
    return ARENAS.get(venue) or OTHER_VENUES[venue]


def distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle (haversine) distance: the shortest path over the Earth's surface."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def utc_offset_hours(time_zone: str, date: pd.Timestamp) -> float:
    """Clock offset from UTC on that date (evening local time), e.g. New York in January = -5."""
    local_evening = datetime(date.year, date.month, date.day, 19, tzinfo=ZoneInfo(time_zone))
    return local_evening.utcoffset().total_seconds() / 3600


def travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: game_id, team, and the TRAVEL_FEATURES columns."""
    games = games.assign(venue=game_venues(games))
    # A team is "on the road" if it is the away team or the game is at a neutral site. (Not
    # "venue != own arena": Toronto's 2020-21 home games in Tampa were still home games.)
    sides = [games[["game_id", "game_date", "season", "venue"]].assign(
                 team=games[f"{side}_team"],
                 on_road=(side == "away") | games["is_neutral_site"].eq(1))
             for side in ["home", "away"]]
    trips = pd.concat(sides, ignore_index=True).sort_values(["team", "game_date"]).reset_index(drop=True)

    info = {venue: venue_info(venue) for venue in trips["venue"].unique()}
    trips["lat"] = trips["venue"].map(lambda v: info[v][0])
    trips["lon"] = trips["venue"].map(lambda v: info[v][1])
    trips["utc_offset"] = [utc_offset_hours(info[v][2], d) for v, d in zip(trips["venue"], trips["game_date"])]
    trips["time_zone"] = trips["venue"].map(lambda v: info[v][2])

    by_team_season = trips.groupby(["team", "season"])
    previous = by_team_season[["lat", "lon", "time_zone"]].shift(1)
    trips["travel_km"] = [
        distance_km(a, b, c, d) if not np.isnan(a) else np.nan
        for a, b, c, d in zip(previous["lat"], previous["lon"], trips["lat"], trips["lon"])
    ]
    # Compare both venues' clocks on the SAME date: otherwise a daylight-saving switch between
    # the two games (which moves every team's clock) would look like an extra hour of travel.
    trips["tz_change_hours"] = [
        offset - utc_offset_hours(previous_zone, date) if isinstance(previous_zone, str) else np.nan
        for offset, previous_zone, date in zip(trips["utc_offset"], previous["time_zone"], trips["game_date"])
    ]

    # Distance over the last 7 days, including the trip to this game (season openers count as 0 km).
    trips["travel_km_last7d"] = (
        trips.assign(km=trips["travel_km"].fillna(0)).set_index("game_date")
        .groupby(["team", "season"])["km"].rolling("7D").sum().reset_index(level=[0, 1], drop=True).to_numpy()
    )
    trips.loc[by_team_season.cumcount() == 0, "travel_km_last7d"] = np.nan

    # Position in the current road trip: count consecutive road games up to and including this one.
    road_run_id = (~trips["on_road"]).groupby([trips["team"], trips["season"]]).cumsum()
    trips["road_trip_game"] = trips.groupby(["team", "season", road_run_id])["on_road"].cumsum().where(trips["on_road"], 0)
    return trips[["game_id", "team", *TRAVEL_FEATURES]]
