# NBA Game Prediction 

([nba-prediction-lemon.vercel.app](url))

Calibrated win probabilities for NBA regular-season games, built only from information available **60 minutes
before tip-off**: team strength (Elo and recent form), rest, home court, and the NBA's **official pre-game injury
reports**. Every modeling decision was made with chronological, walk-forward validation, and the final model was
**pre-registered and tested once** on a held-out season.

## Results

**Final test: 2025-26 season (1,230 games), run once after the model and protocol were locked**

| Model | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| **Final model** (logistic regression, 11 inputs incl. injury features) | 68.8% | **0.5820** | **0.1995** | **0.752** |
| Same model without injury features | 69.2% | 0.5963 | 0.2049 | 0.738 |
| Elo (tuned) | 68.9% | 0.5973 | 0.2051 | 0.738 |
| Always pick the home team | 55.4% | 0.6874 | 0.2471 | 0.500 |

- Log loss vs. tuned Elo: **−0.0154** (95% date-block bootstrap interval −0.0255 to −0.0051).
- Injury features on unseen data: **−0.0143** log loss vs. the same model without them (−0.0228 to −0.0058).
- Calibrated: predicted and actual win rates agree within noise across the range.
- Accuracy rises with confidence: when the model is at least **70%** confident (42% of games) it is right **79.0%**
  of the time; at least **80%** (19% of games): **88.6%**.

**Walk-forward validation (2022-23, 2023-24, 2024-25; 3,690 games, each predicted by a model trained only on earlier
seasons)**: final model accuracy 66.5%, log loss 0.6079, AUC 0.721, calibration slope 1.00.

Full details: [`docs/final_test_protocol.md`](docs/final_test_protocol.md).

## Methodology

**1. Data** (`src/collect_data.py`, `src/clean_data.py`)
- Team box scores for 2018-19 to 2025-26 from the NBA stats API (`nba_api`, `LeagueGameLog`), with retries and caching.
- Cleaned to one row per game, with checks for duplicates, two distinct teams, one winner, and valid scores.
- Neutral-site games (international games, NBA Cup semifinals, the 2020 Orlando bubble) are flagged; the API's
  encoding of them changed in 2024-25, so earlier ones are documented by hand.

**2. Features** (`src/feature_engineering.py` and helpers), all computed from games **before** each game
- Recent form: win % (last 5 / 10), points scored / allowed and point differential (last 10). Shift-then-roll, reset each season.
- Rest: days since the previous game, back-to-backs.
- **Elo ratings** (FiveThirtyEight-style, margin of victory, season carry-over), warmed up on 2018-20.
- **Injury availability** (`src/injury_download.py`, `src/injury_reports.py`, `src/availability_features.py`,
  `src/injury_impact.py`): about 4,400 official NBA injury report PDFs, parsed with `pdfplumber`. For each game, the latest
  report published at least 60 minutes before **its own** tip-off. Features: expected minutes missing (weighted by each
  listed player's historical chance of playing, learned from earlier games), player impact above an average player
  (Game Score per minute, shrunk toward the prior season), and expected stars missing.

**3. Models** (`src/logistic_model.py`, `src/random_forest_model.py`, `src/xgboost_model.py`, `src/elo_baseline.py`)
- Logistic regression on home-minus-away differences, **no intercept**: equal teams at a neutral site get exactly 50%,
  and the home-court flag learns home advantage. Random forest and XGBoost were tuned but never beat it.

**4. Validation** (`src/split.py`, `src/tuning.py`, `src/walk_forward.py`, `src/accuracy_experiments.py`)
- Chronological splits only. Feature groups were added one at a time with walk-forward ablation and date-block
  bootstrap intervals (games on the same night are not independent).
- Kept: injury availability. Rejected (no out-of-sample gain): offensive/defensive ratings, strength of schedule,
  travel distance / time zones / road trips, plus-minus impact, ensembles, Elo re-tuning.

**5. Final test** (`src/final_test.py`)
- Model, inputs, settings and scoring were written down and committed before the test season was evaluated.
  The script refuses to run without a confirmation flag and refuses a second run.

**6. Where is the model reliable?** (`src/category_analysis.py`)
- Calibration checked in 32 game categories (confidence, Elo gap, home/away, injury uncertainty, rest, recent form,
  season stage): no category showed a systematic miscalibration beyond what chance predicts.

## Project structure

```text
nba-prediction/
├── data/                 # raw downloads and processed tables (not committed; rebuilt by the scripts)
├── docs/
│   ├── data_availability_audit.md   # which features were feasible, leakage risks, results of each feature group
│   └── final_test_protocol.md       # pre-registered final test and its results
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── results/              # small committed outputs (settings, comparison tables, charts, final test results)
├── src/                  # pipeline modules (see Methodology)
├── tests/                # pytest: data validation, leakage ("poison") tests, model and pipeline tests
├── pytest.ini
└── requirements.txt
```

## Installation

Requires Python 3.11+ (developed on 3.12).

```bash
git clone https://github.com/z3nsz/nba-prediction.git
cd nba-prediction
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run in order (each step caches its downloads, so re-runs are fast):

```bash
python -m src.collect_data                       # team game logs, 2018-19 to 2025-26
python -m src.clean_data                         # one row per game -> data/processed/nba_games.csv
python -m src.feature_engineering                # pre-game features -> data/processed/nba_features.csv
python -m src.injury_download                    # official injury reports, development seasons (~1 hour, ~200 MB)
python -m src.availability_features              # injury availability features
python -m src.injury_impact                      # probability of playing and player impact features
python -m src.walk_forward                       # walk-forward feature-group ablation
python -m src.accuracy_experiments               # final-model experiments vs. the 10-input baseline
python -m src.category_analysis                  # accuracy and calibration by game category
python -m pytest                                 # tests
```

Test-season inputs are built with `--test-season` (`src.injury_download`, `src.availability_features`,
`src.injury_impact`). The one-time final test has already been run; `src/final_test.py` refuses to run again.

## Data sources

- **NBA statistics:** [`nba_api`](https://github.com/swar/nba_api) (stats.nba.com): team and player game logs.
- **Injury reports:** the NBA's official pre-game injury report PDFs (`ak-static.cms.nba.com/referee/injury/`).
- **Elo method:** settings follow FiveThirtyEight's published NBA Elo (re-tuned on training seasons for the baseline).

Data files are not committed; the scripts above rebuild them.

## Limitations

- Predictions are made 60 minutes before tip-off; later news (late scratches, starting lineups) is not used.
- Box-score player impact (Game Score) undervalues defense-first players.
- Early-season estimates rely on few games.
- 2025-26 can no longer serve as an untouched test set; future model changes need a new season to evaluate.
