"""Parse the NBA's official pre-game injury report PDFs (prototype, Step 3.2c).

Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>PM.pdf
Reports are published several times a day (3 per day in 2020-21, hourly from
2022-23, every 30 minutes in 2025-26). Each report lists, for every upcoming
game: date, tip-off time (ET), matchup, team, player, status, reason.

Parsing strategy: the PDFs come in two layouts with different column
positions, so the column boundaries are read from each page's header row.
Every word is assigned to a column by its x position, words are grouped into
lines by their y position, and a line counts as a player row only if its
status column holds a known status. Lines that are only a wrapped "Reason"
have no status and are skipped. Game date, time, matchup and team are printed
only on the first row of each group, so they are carried forward.

Downloaded PDFs are data, not code: they are only read, never executed.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
import pdfplumber

from src.collect_data import RAW_DATA_DIR

INJURY_REPORT_DIR = RAW_DATA_DIR / "injury_reports"
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
COLUMNS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]
# First header word of each column, in order. "Game" appears twice (Date, Time).
HEADER_STARTS = ["Game", "Game", "Matchup", "Team", "Player", "Current", "Reason"]
PAGE_FOOTER_PATTERN = re.compile(r"\bPage \d+ of \d+\b")
REPORT_TIME_PATTERN = re.compile(r"Injury Report: (\d{2}/\d{2}/\d{2}) (\d{2}:\d{2} [AP]M)")


def column_starts(words: list[dict]) -> list[float] | None:
    """x position where each column starts, read from the page's header row (None if no header)."""
    matchup = next((w for w in words if w["text"] == "Matchup"), None)
    if matchup is None:
        return None
    header = sorted((w for w in words if abs(w["top"] - matchup["top"]) < 2), key=lambda w: w["x0"])
    starts, wanted = [], list(HEADER_STARTS)
    for word in header:
        if wanted and word["text"] == wanted[0]:
            starts.append(word["x0"])
            wanted.pop(0)
    return starts if not wanted else None


def page_lines(page, previous_starts: list[float] | None) -> tuple[str | None, list[float] | None, list[dict]]:
    """Report timestamp, column positions, and the page's lines split into columns.

    Pages without their own header row reuse the previous page's column positions.
    """
    words = page.extract_words(x_tolerance=1.5)
    text = " ".join(w["text"] for w in words)
    stamp = REPORT_TIME_PATTERN.search(text)
    own_starts = column_starts(words)
    starts = own_starts or previous_starts
    if starts is None:
        return (stamp.group(0) if stamp else None), None, []

    header_top = next(w for w in words if w["text"] == "Matchup")["top"] if own_starts else -1
    lines: dict[int, dict[str, list[str]]] = {}
    for word in words:
        if word["top"] <= header_top + 2 or "Injury Report:" in word["text"]:
            continue  # title and header rows
        column_index = max(i for i, start in enumerate(starts) if word["x0"] >= start - 2) if word["x0"] >= starts[0] - 2 else 0
        line_key = round(word["top"] / 3)  # words within ~3 points vertically share a line
        lines.setdefault(line_key, {c: [] for c in COLUMNS})[COLUMNS[column_index]].append(word["text"])
    ordered = [{c: " ".join(parts).strip() for c, parts in lines[k].items()} for k in sorted(lines)]
    return (stamp.group(0) if stamp else None), starts, ordered


def parse_report(path: Path) -> pd.DataFrame:
    """One row per player listed in the report, plus rows for teams marked NOT YET SUBMITTED."""
    rows, report_stamp, starts = [], None, None
    carried = {"game_date": None, "game_time": None, "matchup": None, "team": None}
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            stamp, starts, lines = page_lines(page, starts)
            report_stamp = report_stamp or stamp
            for line in lines:
                line_text = " ".join(line.values())
                # Skip titles and "Page 3 of 9" footers. A footer must not be carried forward as a
                # team name: the players at the top of the next page belong to the previous team.
                if "Injury Report:" in line_text or PAGE_FOOTER_PATTERN.search(line_text):
                    continue
                for field in carried:
                    if line[field]:
                        carried[field] = line[field]
                if line["status"] in STATUSES:
                    rows.append({**carried, "player": line["player"], "status": line["status"]})
                elif "NOT YET SUBMITTED" in line["reason"].replace(" ", "").upper().replace("NOTYETSUBMITTED", "NOT YET SUBMITTED"):
                    rows.append({**carried, "player": None, "status": "Not yet submitted"})

    report = pd.DataFrame(rows, columns=["game_date", "game_time", "matchup", "team", "player", "status"])
    match = REPORT_TIME_PATTERN.search(report_stamp or "")
    report["report_time"] = pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%m/%d/%y %I:%M %p") if match else pd.NaT
    report["tip_off"] = pd.to_datetime(report["game_date"] + " " + report["game_time"].str.replace(r"\s*\(ET\)", "", regex=True),
                                       format="%m/%d/%Y %I:%M", errors="coerce")
    # Game times are printed without AM/PM, and "12:30" parses as 00:30. NBA games tip between
    # 11 AM and 11 PM ET, so hours 0-10 are afternoon/evening: add 12 (12:30 -> 12:30 PM, 7:00 -> 7 PM).
    is_afternoon_or_evening = report["tip_off"].dt.hour.between(0, 10)
    report.loc[is_afternoon_or_evening, "tip_off"] += pd.Timedelta(hours=12)
    report["source_file"] = path.name
    return report


def normalize_player_name(name: str) -> str:
    """Comparable key for a player name in either format.

    'Harper Jr., Ron' (injury report) and 'Ron Harper Jr.' (box scores) both become
    'ronharperjr': reorder "Last, First", strip accents, keep letters only.
    """
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", name.lower())
