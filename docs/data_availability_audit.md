# Data Availability Audit — New Pre-Game Features (Step 3.1)

**Goal:** decide which new features can be built **reliably and without leakage** before writing code.
**Rule:** a feature is only usable if its value was publicly known **before tip-off** for every game we model.
**Prediction timestamp:** features are defined as known **before tip-off**. Where a source is only known
shortly before tip-off (inactive lists), that is noted, because a prediction may be needed earlier.

## What our existing data already contains

`data/raw/team_game_logs_*.csv` (LeagueGameLog, team mode), 8 seasons, 2018-19 to 2025-26:
FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS, plus-minus, minutes
(240 = regulation, 265/290/315/340 = 1-4 overtimes). Only one missing value (an undefined 0/0 FT%).

That is enough to estimate **possessions** for every game, so offensive, defensive and net ratings
can be built from data we already have. Opponent Elo comes from our own pre-game Elo ratings.

## Sources checked (Oct 2026)

| Source | Checked how | What we found |
|---|---|---|
| Team game logs (have) | Inspected all 8 seasons | Complete; has every input for possessions |
| Our Elo ratings (have) | `compute_elo()` | Pre-game rating for every team in every game |
| NBA official injury report PDFs (`ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf`) | HTTP requests for 8 dates, Dec 2020 to Dec 2025 | **All 8 exist.** ~85 KB each. The Jan 15 2024 "5 PM" report was last modified 5:31 PM ET: before evening games, **after** afternoon tip-offs |
| Inactive player lists (`BoxScoreSummaryV2`, nba_api) | Called for one 2020-21 and one 2024-25 game | Works. Player ID, name, team. **1 call per game** (~9,500 total). nba_api warns data may be missing on or after 2025-04-10 (V3 endpoint needed) |
| Player game logs (`LeagueGameLog`, player mode) | Called for 2024-25 | Works. 26,306 rows, **1 call per season**. Minutes, points, +/- per player per game. Lists **only players who played** |
| Projected starting lineups | No official historical archive found | Third-party projections exist but are not archived publicly for past games |
| Arena locations / time zones | Not an API: fixed public facts | City coordinates and time zones are well documented; must handle neutral sites and the Raptors' 2020-21 season in **Tampa** |

## Feature availability table

| Feature group | Feature | Source | Coverage | Leakage risk | Difficulty | Decision |
|---|---|---|---|---|---|---|
| **1. Ratings** | Offensive / defensive / net rating (last 5, 10) | Team game logs (have) | All seasons | Low (shift before rolling) | Low | **Build now** |
| **2. Schedule strength** | Avg opponent Elo (last 5, 10) | Our pre-game Elo | All seasons | Low (opponent's rating *before* that earlier game) | Low | **Build now** |
| 2. Schedule strength | Opponent-adjusted form: margin minus Elo-expected margin (last 10) | Elo + scores (have) | All seasons | Low | Low | **Build now** |
| **3. Travel & fatigue** | Back-to-back, days of rest | Schedule (have) | All seasons | None | — | Already built |
| 3. Travel & fatigue | Consecutive road games | Schedule (have) | All seasons | None | Low | Build next |
| 3. Travel & fatigue | Travel distance since previous game | Schedule + fixed arena coordinates | All seasons | None | Medium: need venue table incl. neutral sites + 2020-21 Tampa | Build next |
| 3. Travel & fatigue | Time-zone change | Fixed time-zone table | All seasons | None | Medium: Arizona has no daylight saving | Build next, documented |
| **4. Availability** | Players listed OUT / doubtful / questionable | Official injury report PDFs | 2020-21 → 2025-26 (verified 8 dates) | **Medium-high:** must pick the last report published **before each game's tip-off**, which needs tip times; parsing PDFs; matching names to player IDs | High | **Prototype first** (needs your OK to download PDFs) |
| 4. Availability | Players inactive for the game | `BoxScoreSummaryV2` | 2020-21 → Apr 2025 confirmed; later needs V3 | **Medium:** known ~30 min before tip, later than the 60-minute prediction time | Medium: ~9,500 cached API calls | Candidate, after injury-report check |
| **5. Lineups / minutes** | Expected minutes per player from prior games | Player game logs | All seasons | Low if shifted (prior games only) | Medium | Build together with group 4 |
| 5. Lineups / minutes | Team strength adjusted for missing players | Expected minutes × availability (group 4) | Depends on group 4 | Inherits group 4's risk | High | Only if group 4 is reliable |
| 5. Lineups / minutes | Projected starters | None found | — | — | — | **Not feasible** (no reliable archive) |
| 5. Lineups / minutes | Actual starters / minutes in the game | Box scores | — | **Leakage:** only known after the game | — | **Never use** |

## Key leakage traps identified

1. **Box scores list only players who played.** "Missing from this game's box score" includes coach's
   decisions and in-game injuries. That is post-game information and must never be used as availability.
2. **Injury report timing.** Reports are published several times a day; using a report issued after an
   afternoon tip-off would leak late scratches. Each game needs the last report before *its* tip-off.
3. **Opponent Elo for schedule strength** must be the opponent's rating *before* the earlier game,
   which is what our Elo table stores.
4. **Rolling ratings** must exclude the current game (shift before rolling), exactly like Phase 1 features.

## Evaluation protocol for new features (Step 3.3)

The 2023-25 validation seasons were already used heavily for model selection in Phase 2, so new
feature decisions use **walk-forward validation inside the development seasons (2020-25)**:

| Fold | Train on | Evaluate on |
|---|---|---|
| 1 | 2020-21, 2021-22 | 2022-23 |
| 2 | 2020-21 → 2022-23 | 2023-24 |
| 3 | 2020-21 → 2023-24 | 2024-25 |

The 2025-26 test season is not used for any decision, and no score on it is computed or reported.

## Priority order

1. Ratings + schedule strength (existing data, low risk) ← Step 3.2a
2. Travel & fatigue extensions (fixed tables, low risk) ← Step 3.2b
3. Injury report prototype: download a handful of PDFs, check format, timing, and parsing reliability
   on a sample before committing to the full build ← needs approval to download
4. Expected minutes + availability-adjusted strength, only if step 3 proves reliable

## Injury report prototype results (Step 3.2c, 10 sample PDFs, development seasons only)

Samples: Jan 2021, and Martin Luther King Jr. Day 2022-2025 (chosen because it has afternoon tip-offs).
Downloaded with approval; 470 KB, stored in git-ignored `data/raw/injury_reports/`.

| Check | Result |
|---|---|
| Publishing schedule | 2020-21: 3 reports/day (1, 5, 8 PM). 2022-23 to 2024-25: hourly, 11 AM-9 PM. 2025-26: every 30 min, new filename format (`_05_00PM`) |
| Layouts | Two (2020-23 and 2024+), with different column positions; parser reads positions from each page's header |
| Report timestamp | Printed inside each report (a file named "11AM" is the 11:30 AM report) |
| Tip-off time per game | Printed for every game, so reports can be matched to games without another source |
| Same-day games found in our games table | **100%** in all 10 reports |
| Rows whose team belongs to their matchup | **100%** of 1,053 (after fixing a page-footer bug) |
| Player names matched to exactly one player ID | **94.9%** (0 ambiguous). Most misses are players who sat out the whole season (no game logs); a few are name variants (e.g. "Nicolas" vs "Nic") |
| Reports published **after** a same-day game's tip-off | **25 of 98** game listings: confirms the timing trap |

Bugs found and fixed during the prototype:
1. Pages after the first have no header row, so they were skipped: column positions are now reused.
2. "Page 1 of 9" footers were carried forward as team names (71 rows): footers are now skipped.
3. Noon tip-offs ("12:30", printed without AM/PM) parsed as 00:30: hours 0-10 now map to PM.

**Verdict: feasible.** Rule for a leakage-safe feature: for each game, use the latest report published at
least a fixed buffer (e.g. 60 minutes) before that game's tip-off. Full build estimate for the development
seasons: roughly 2,000-2,500 PDFs (one per distinct tip-off slot per game day), on the order of 100 MB,
which needs a separate download approval.

## Walk-forward result: ratings and schedule strength (Step 3.3, part 1)

Folds: 2020-22 → 2022-23, 2020-23 → 2023-24, 2020-24 → 2024-25 (3,690 games). Test season not used.

| Logistic regression | Pooled log loss | Change vs. Phase 2 | 95% interval (date blocks) |
|---|---|---|---|
| Phase 2 (7 inputs) | 0.6190 | — | — |
| + ratings | 0.6196 | +0.0006 | [−0.0006, +0.0019] |
| + schedule strength | 0.6194 | +0.0004 | [−0.0005, +0.0012] |
| + both | 0.6201 | +0.0011 | [−0.0003, +0.0026] |

**Decision: do not add these groups.** Neither improves log loss in any fold, and adding both makes
random forest and XGBoost worse too. They are different measures of team strength, which Elo and recent
point differential already capture. The features stay in the dataset (documented, tested) but are not
model inputs.

## Full build and walk-forward result: player availability (Steps 3.2c / 3.3, part 2)

**Download:** every development-season game got the latest official report printed at least 60 minutes before
its tip-off (`src/injury_download.py`; 39 noon/early games use the previous evening's last report).
3,537 PDFs, 195.6 MB, git-ignored. This was about twice the ~100 MB estimate given before approval.

**Features** (`src/availability_features.py`), per team: expected minutes of players listed Out/Doubtful,
expected minutes of Questionable players, and starters (28+ minutes) out. Expected minutes = season average
from games on earlier dates only. Coverage: 99.8% of team-games; the rest are NaN (unknown), never guessed.

Issues found and fixed:
1. Leakage guard fired: 3 games' tip-off times changed during the day (e.g. 8:30 PM → 7:30 PM). The earlier
   time is assumed and those games are marked unknown.
2. Teams marked NOT YET SUBMITTED were counted as "no injuries" (0) because of a `~` on Python booleans
   (12 team-games, all 2020-21). They are now unknown (NaN).

| Logistic regression, walk-forward (3,690 games) | Log loss | Accuracy | AUC | Calibration error |
|---|---|---|---|---|
| Phase 2 (7 inputs) | 0.6190 | 65.0% | 0.705 | 0.020 |
| + availability | **0.6097** | **66.9%** | **0.719** | **0.012** |

Change: **−0.0093** log loss, 95% interval [−0.0142, −0.0044] (date-block bootstrap); better in all 3 seasons.
Random forest and XGBoost also improve with availability but stay behind logistic regression.

**Decision: add the availability group.** Learned weights are positive and sensible (questionable players
count about a third as much as Out players).

**Limitations**
- Features assume a prediction made 60 minutes before tip-off. Any forecast the model is compared with must be
  taken at the same time or later, otherwise the model would know injury news the other forecast had not seen yet.
- Early-season expected minutes are noisy (few games played).
- For the final 2025-26 evaluation, that season's reports (new 30-minute schedule and `_05_00PM` filenames) and
  player logs are needed. Downloading them needs approval, and the parser must be re-checked on the new format.

## Walk-forward result: travel and fatigue (Step 3.2b)

Features (`src/travel_features.py`): great-circle km from the previous venue, km over the last 7 days,
absolute time-zone change, and position in the current road trip. Venues handle neutral sites, the 2019-20
Orlando bubble, and Toronto's 2020-21 home games in Tampa; time zones are looked up per date.

Bugs found and fixed:
1. Toronto's entire 2020-21 season counted as one 72-game road trip ("venue is not own arena"). Road games are
   now defined by the game itself: away team or neutral site.
2. A daylight-saving change between two games was counted as an extra hour of travel (NY → LA showed 4 hours).
   Both venues are now compared on the same date.

| Logistic regression | Log loss | Change vs. Phase 2 | 95% interval | Calibration error |
|---|---|---|---|---|
| + travel | 0.6204 | +0.0014 | [−0.0004, +0.0032] | 0.021 |
| + availability | 0.6097 | −0.0093 | [−0.0142, −0.0044] | 0.012 |
| + availability + travel | 0.6109 | −0.0081 | [−0.0132, −0.0030] | 0.020 |

**Decision: do not add travel.** It barely overlaps with back-to-backs or rest (correlations ≤ 0.16) yet adds
no predictive value, and it worsens calibration. The away team almost always traveled, so `home_court` already
captures most of it. Back-to-back (already in the model) remains the fatigue signal that matters.

## Walk-forward result: player impact and ensembles (accuracy push)

Missing players weighted by impact instead of minutes: season-to-date average **Game Score** (Hollinger) and
average **plus-minus**, from earlier dates only. Also tested: averaging logistic regression and XGBoost.

| Model (walk-forward, 3,690 games) | Log loss | Accuracy | AUC | Calibration error | vs. logistic + availability |
|---|---|---|---|---|---|
| Logistic + availability (current) | 0.6097 | 66.9% | 0.719 | 0.012 | — |
| Logistic + availability + Game Score | 0.6086 | 67.2% | 0.721 | 0.025 | −0.0011 [−0.0029, +0.0007] |
| Logistic + availability + plus-minus | 0.6097 | 67.0% | 0.719 | 0.015 | +0.0001 [−0.0003, +0.0004] |
| Ensemble (logistic + XGBoost) + availability | 0.6113 | 66.9% | 0.717 | 0.017 | +0.0016 [+0.0001, +0.0031] (worse) |

**Decision: keep logistic + availability.** Game Score improves log loss and accuracy slightly in 2 of 3 seasons,
but the improvement is within noise (interval includes 0; accuracy noise is about ±0.8 points on 3,690 games) and
calibration error doubles. Plus-minus adds nothing; the ensemble is worse. Accuracy is close to the level a
perfectly calibrated model with these probabilities would reach (66.2% expected vs. 66.9% observed).

## Accuracy experiments toward 70% (`python -m src.accuracy_experiments`)

Walk-forward, 3,690 games (2022-23, 2023-24, 2024-25), logistic regression C = 1.0, test season not used.
Baseline = locked 10-input model. Calibration slope (1.0 = perfect) was ~1.00 for every variant.

| Experiment | Accuracy | Log loss | Brier | AUC | vs. baseline (95% date-block) | Seasons better |
|---|---|---|---|---|---|---|
| Baseline (10 inputs) | 66.9% | 0.6097 | 0.2111 | 0.719 | — | — |
| E1a: Questionable minutes × P(miss) | 67.1% | 0.6091 | 0.2108 | 0.719 | −0.0006 [−0.0011, −0.0001] | 3/3 |
| E1b: all listed minutes × P(miss) | 66.9% | 0.6089 | 0.2108 | 0.720 | −0.0007 [−0.0016, +0.0002] | 3/3 |
| E2a: + value above average | 66.7% | 0.6087 | 0.2106 | 0.720 | −0.0010 [−0.0026, +0.0005] | 3/3 |
| E2b: + value + stars missing | 66.7% | 0.6087 | 0.2107 | 0.720 | −0.0010 [−0.0025, +0.0006] | 3/3 |
| **E1+E2: P(miss) minutes + value + stars** | 66.5% | **0.6079** | **0.2103** | **0.721** | **−0.0018 [−0.0037, −0.0000]** | 3/3 |
| E3: Elo re-tuned per fold | 67.0% | 0.6097 | 0.2110 | 0.718 | +0.0000 [−0.0014, +0.0014] | 1/3 |

Findings: Questionable players play 50.4% of the time, Doubtful 2.1%, Probable 90.3%; individual histories vary a lot
(e.g. 97% vs. 0% for two frequently-listed players). Player impact (Game Score per minute above league average, shrunk
toward last season) ranks stars correctly but misses defense. Elo re-tuning chose carry-over 0.5 in every fold and
added nothing. All-game accuracy did not move beyond noise (±0.8 points); 70% was not reached.
Caveat: seven variants were compared, so the borderline E1+E2 interval should be read with that in mind.
