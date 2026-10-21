import { FACTORS, Game, TEAMS, loadData } from "../lib/data";

const pct = (p: number) => `${Math.round(p * 100)}%`;

function tipTime(iso: string) {
  const [, time] = iso.split("T");
  const [h, m] = time.split(":").map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"} ET`;
}

function longDate(date: string) {
  return new Date(`${date}T12:00:00`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
}

function reportTime(stamp: string | null) {
  if (!stamp) return "no injury report yet";
  const [, hh, mm, ampm] = stamp.match(/_(\d\d)_(\d\d)(AM|PM)$/) ?? [];
  return `injury report ${Number(hh)}:${mm} ${ampm}`;
}

function topFactors(game: Game) {
  return Object.entries(game.contributions)
    .filter(([, v]) => Math.abs(v) >= 0.01)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([name, v]) => ({ name: FACTORS[name] ?? name, team: v > 0 ? game.home : game.away }));
}

function GameCard({ game }: { game: Game }) {
  const done = game.home_score !== null && game.away_score !== null;
  const homeWon = done && game.home_score! > game.away_score!;
  const pick = game.p_home >= 0.5 ? game.home : game.away;
  const correct = done && (game.p_home >= 0.5) === homeWon;
  return (
    <article className="card">
      <div className="card-top">
        <span className="time">{tipTime(game.tip_off_et)}</span>
        {done ? (
          <span className={`badge ${game.final ? (correct ? "good" : "bad") : ""}`}>
            {game.final ? (correct ? "Correct" : "Missed") : "Not counted"}
          </span>
        ) : (
          <span className={`badge ${game.final ? "locked" : ""}`}>{game.final ? "Locked" : "Preview"}</span>
        )}
      </div>
      {[{ team: game.away, p: 1 - game.p_home, score: game.away_score, tag: "Away" },
        { team: game.home, p: game.p_home, score: game.home_score, tag: "Home" }].map((side) => (
        <div className={`team ${side.team === pick ? "pick" : ""}`} key={side.team}>
          <div className="team-name">
            <span className="abbr">{side.team}</span>
            <span className="full">{TEAMS[side.team] ?? side.team}</span>
            <span className="tag">{side.tag}</span>
          </div>
          <div className="team-right">
            {done && <span className="score">{side.score}</span>}
            <span className="prob">{pct(side.p)}</span>
          </div>
        </div>
      ))}
      <div className="bar" aria-hidden>
        <div className="bar-away" style={{ width: pct(1 - game.p_home) }} />
      </div>
      <ul className="factors">
        {topFactors(game).map((f) => (
          <li key={f.name}>
            <span>{f.name}</span>
            <span className="favors">favors {f.team}</span>
          </li>
        ))}
      </ul>
      <p className="source">{reportTime(game.report)}</p>
    </article>
  );
}

export default function Home() {
  const data = loadData();
  const dates = [...new Set(data.games.map((g) => g.date))].sort().reverse();
  const latest = dates[0];
  const record = data.record;

  return (
    <main>
      <header>
        <p className="eyebrow">NBA {data.season} · updated {data.updated_et.replace("T", " ")} ET</p>
        <h1>NBA Game Predictions</h1>
        <p className="lede">
          Win probabilities for every regular-season game, made by a logistic regression model trained on{" "}
          {data.model.training_games.toLocaleString()} games ({data.model.trained_on[0]} to{" "}
          {data.model.trained_on.at(-1)}). The model uses team strength, recent form, rest and official NBA
          injury reports.
        </p>
      </header>

      <section className="stats">
        <div className="stat">
          <span className="stat-value">{record.games ? pct(record.accuracy!) : "—"}</span>
          <span className="stat-label">Accuracy{record.games ? ` (${record.correct} of ${record.games})` : ""}</span>
        </div>
        <div className="stat">
          <span className="stat-value">{record.games ? record.log_loss!.toFixed(3) : "—"}</span>
          <span className="stat-label">Log loss (lower is better)</span>
        </div>
        <div className="stat">
          <span className="stat-value">{record.games ? record.brier!.toFixed(3) : "—"}</span>
          <span className="stat-label">Brier score (lower is better)</span>
        </div>
      </section>
      {record.games > 0 && record.games < 100 && (
        <p className="note">
          Only {record.games} games so far: early numbers swing a lot. On a full held-out season (2025-26, tested
          once, before this season) the model was right 68.8% of the time.
        </p>
      )}

      {dates.map((date) => (
        <section key={date}>
          <h2>
            {date === latest ? "Latest games · " : ""}
            {longDate(date)}
          </h2>
          <div className="grid">
            {data.games.filter((g) => g.date === date).map((g) => <GameCard game={g} key={g.id} />)}
          </div>
        </section>
      ))}

      <footer>
        <p>
          <strong>How it works.</strong> Each morning the job publishes a preview. Sixty minutes before tip-off the
          prediction is locked using the latest official injury report, and only locked predictions count in the
          track record.
        </p>
        <p>For learning and research only, not betting advice.</p>
      </footer>
    </main>
  );
}
