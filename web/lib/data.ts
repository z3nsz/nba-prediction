import { readFileSync } from "fs";
import path from "path";

export type Game = {
  id: string;
  date: string;
  tip_off_et: string;
  home: string;
  away: string;
  p_home: number;
  final: boolean;
  report: string | null;
  home_score: number | null;
  away_score: number | null;
  contributions: Record<string, number>;
};

export type Data = {
  updated_et: string;
  season: string;
  model: { description: string; training_games: number; trained_on: string[] };
  record: { games: number; correct?: number; accuracy?: number; log_loss?: number; brier?: number };
  games: Game[];
};

// Written by the daily job (src/live/predict.py); read at build time, so each new commit redeploys the site.
export function loadData(): Data {
  const file = path.join(process.cwd(), "public", "data", "predictions.json");
  return JSON.parse(readFileSync(file, "utf8"));
}

export const TEAMS: Record<string, string> = {
  ATL: "Atlanta Hawks", BOS: "Boston Celtics", BKN: "Brooklyn Nets", CHA: "Charlotte Hornets",
  CHI: "Chicago Bulls", CLE: "Cleveland Cavaliers", DAL: "Dallas Mavericks", DEN: "Denver Nuggets",
  DET: "Detroit Pistons", GSW: "Golden State Warriors", HOU: "Houston Rockets", IND: "Indiana Pacers",
  LAC: "LA Clippers", LAL: "Los Angeles Lakers", MEM: "Memphis Grizzlies", MIA: "Miami Heat",
  MIL: "Milwaukee Bucks", MIN: "Minnesota Timberwolves", NOP: "New Orleans Pelicans", NYK: "New York Knicks",
  OKC: "Oklahoma City Thunder", ORL: "Orlando Magic", PHI: "Philadelphia 76ers", PHX: "Phoenix Suns",
  POR: "Portland Trail Blazers", SAC: "Sacramento Kings", SAS: "San Antonio Spurs", TOR: "Toronto Raptors",
  UTA: "Utah Jazz", WAS: "Washington Wizards",
};

// Plain-English names for the model inputs.
export const FACTORS: Record<string, string> = {
  elo_diff: "Team strength (Elo)",
  home_court: "Home court",
  point_diff_last10_diff: "Point margin, last 10",
  win_pct_last10_diff: "Win %, last 10",
  win_pct_last5_diff: "Win %, last 5",
  rest_days_diff: "Rest days",
  back_to_back_diff: "Back-to-back",
  exp_missing_minutes_diff: "Injuries: missing minutes",
  starters_out_diff: "Injuries: starters out",
  exp_missing_value_diff: "Injuries: missing production",
  exp_stars_missing_diff: "Injuries: stars out",
};
