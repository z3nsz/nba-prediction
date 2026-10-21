import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Details, FACTORS, Game, Side, TEAMS, loadData, loadDetails, reportUrl } from "../../../lib/data";
import { Logo, longDate, pct, reportTime, tipTime } from "../../../lib/ui";

type Params = { params: Promise<{ id: string }> };

// One page per game, built from predictions.json at deploy time.
export const dynamicParams = false;

export function generateStaticParams() {
  return loadData().games.map((g) => ({ id: g.id }));
}

function findGame(id: string): Game | undefined {
  return loadData().games.find((g) => g.id === id);
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const game = findGame((await params).id);
  return { title: game ? `${game.away} @ ${game.home} · NBA Game Predictions` : "NBA Game Predictions" };
}

const dash = "—";
const fixed = (v: number | null | undefined, digits = 0) => (v === null || v === undefined ? dash : v.toFixed(digits));
const signed = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? dash : `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
const share = (v: number | null | undefined) => (v === null || v === undefined ? dash : pct(v));

function Factors({ game }: { game: Game }) {
  const rows = Object.entries(game.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const biggest = Math.max(...rows.map(([, v]) => Math.abs(v)), 0.01);
  const total = rows.reduce((sum, [, v]) => sum + v, 0);
  return (
    <section className="panel">
      <h2>Why the model thinks so</h2>
      <p className="muted">
        The model starts at 50/50 and each factor pushes toward one team. Bars are in log-odds, the model&apos;s
        internal scale: they add up to {signed(total, 2)}, which becomes {pct(game.p_home)} for {game.home}.
      </p>
      <div className="factor-head">
        <span className="factor-sides">
          <span>← {game.away}</span>
          <span>{game.home} →</span>
        </span>
      </div>
      <ul className="factor-list">
        {rows.map(([name, v]) => (
          <li key={name} className={Math.abs(v) < 0.005 ? "zero" : ""}>
            <span className="factor-name">{FACTORS[name] ?? name}</span>
            <span className="diverge">
              <span className="half left">
                {v < 0 && <span className="fill away" style={{ width: `${(-v / biggest) * 100}%` }} />}
              </span>
              <span className="half right">
                {v > 0 && <span className="fill home" style={{ width: `${(v / biggest) * 100}%` }} />}
              </span>
            </span>
            <span className="factor-value">{signed(v, 2)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function TeamTable({ details }: { details: Details }) {
  const { home, away } = details;
  const noGames = home.win_pct_last10 === null && away.win_pct_last10 === null;
  const row = (label: string, a: string, h: string) => (
    <tr>
      <th scope="row">{label}</th>
      <td>{a}</td>
      <td>{h}</td>
    </tr>
  );
  return (
    <section className="panel">
      <h2>Team strength and form</h2>
      <table className="compare">
        <thead>
          <tr>
            <th />
            <th><Logo team={away.team} size={22} /> {away.team}</th>
            <th><Logo team={home.team} size={22} /> {home.team}</th>
          </tr>
        </thead>
        <tbody>
          {row("Elo rating", fixed(away.elo), fixed(home.elo))}
          {row("Win %, last 10", share(away.win_pct_last10), share(home.win_pct_last10))}
          {row("Win %, last 5", share(away.win_pct_last5), share(home.win_pct_last5))}
          {row("Point margin, last 10", signed(away.point_diff_last10), signed(home.point_diff_last10))}
          {row("Days since last game", fixed(away.rest_days), fixed(home.rest_days))}
          {row("Back-to-back", away.back_to_back ? "Yes" : "No", home.back_to_back ? "Yes" : "No")}
        </tbody>
      </table>
      {details.elo_home_win_prob !== null && (
        <p className="muted">
          Elo alone (with its home-court bonus) gives {home.team} {pct(details.elo_home_win_prob)}. Elo carries over
          from last season, pulled 25% back toward average each summer.
        </p>
      )}
      {noGames && <p className="muted">No games played yet this season, so recent form and rest count as even.</p>}
    </section>
  );
}

function Injuries({ side }: { side: Side }) {
  return (
    <div className="injury-team">
      <h3><Logo team={side.team} size={22} /> {TEAMS[side.team] ?? side.team}</h3>
      {!side.report_known ? (
        <p className="muted">Report not submitted in time: injuries count as unknown (even).</p>
      ) : side.injuries.length === 0 ? (
        <p className="muted">Nobody listed.</p>
      ) : (
        <table className="injuries">
          <thead>
            <tr>
              <th>Player</th>
              <th>Status</th>
              <th title="Season average minutes before this game">Min/game</th>
              <th title="Chance he plays, from how often players with this status played">Plays</th>
              <th title="Minutes per game times chance he misses">Exp. missing</th>
            </tr>
          </thead>
          <tbody>
            {side.injuries.map((p) => (
              <tr key={p.player}>
                <td>{p.player.split(", ").reverse().join(" ")}</td>
                <td><span className={`status s-${p.status.toLowerCase().replace(/\s+/g, "-")}`}>{p.status}</span></td>
                <td>{fixed(p.avg_minutes, 1)}</td>
                <td>{share(p.p_play)}</td>
                <td>{fixed(p.exp_missing_minutes, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default async function GamePage({ params }: Params) {
  const { id } = await params;
  const game = findGame(id);
  if (!game) notFound();
  const details = loadDetails(id);
  const done = game.home_score !== null && game.away_score !== null;
  const homeWon = done && game.home_score! > game.away_score!;
  const correct = done && (game.p_home >= 0.5) === homeWon;
  const noMinutes = details && [...details.home.injuries, ...details.away.injuries].every((p) => !p.avg_minutes);

  return (
    <main>
      <Link href="/" className="back">← All games</Link>
      <p className="eyebrow">{longDate(game.date)} · {tipTime(game.tip_off_et)}</p>

      <section className="matchup">
        {[{ team: game.away, p: 1 - game.p_home, score: game.away_score, tag: "Away" },
          { team: game.home, p: game.p_home, score: game.home_score, tag: "Home" }].map((side) => (
          <div className="matchup-team" key={side.team}>
            <Logo team={side.team} size={64} />
            <span className="matchup-abbr">{side.team}</span>
            <span className="matchup-tag">{side.tag}</span>
            <span className="matchup-prob">{pct(side.p)}</span>
            {done && <span className="matchup-score">{side.score}</span>}
          </div>
        ))}
      </section>
      <div className="bar big" aria-hidden>
        <div className="bar-away" style={{ width: pct(1 - game.p_home) }} />
      </div>
      <p className="verdict">
        {done
          ? game.final
            ? correct ? "✓ The model picked the winner." : "✗ The model picked the loser."
            : "Finished, but no locked prediction: not counted in the track record."
          : game.final
            ? "Locked: this prediction will not change."
            : "Preview: updates until 60 minutes before tip-off."}
      </p>

      <Factors game={game} />

      {details ? (
        <>
          <TeamTable details={details} />
          <section className="panel">
            <h2>Injury report</h2>
            <p className="muted">
              {details.report ? (
                <>
                  From the official NBA{" "}
                  <a href={reportUrl(details.report)} target="_blank" rel="noreferrer">{reportTime(details.report)}</a>
                </>
              ) : "No official injury report was available yet"}
              {details.cutoff_et && <> (cutoff {tipTime(details.cutoff_et)}, one hour before tip-off)</>}.
              {noMinutes && " Season averages start at zero until players log minutes this season."}
            </p>
            <div className="injury-grid">
              <Injuries side={details.away} />
              <Injuries side={details.home} />
            </div>
          </section>
          <p className="muted small">Prediction made {details.computed_et.replace("T", " ")} ET.</p>
        </>
      ) : (
        <p className="muted">Details for this game are not available yet.</p>
      )}
    </main>
  );
}
