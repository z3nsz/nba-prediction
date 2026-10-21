import { logoUrl } from "./data";

export const pct = (p: number) => `${Math.round(p * 100)}%`;

export function tipTime(iso: string) {
  const [, time] = iso.split("T");
  const [h, m] = time.split(":").map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"} ET`;
}

export function longDate(date: string) {
  return new Date(`${date}T12:00:00`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
}

export function reportTime(stamp: string | null) {
  if (!stamp) return "no injury report yet";
  const [, hh, mm, ampm] = stamp.match(/_(\d\d)_(\d\d)(AM|PM)$/) ?? [];
  return `injury report ${Number(hh)}:${mm} ${ampm}`;
}

export function Logo({ team, size = 28 }: { team: string; size?: number }) {
  // Dark-mode version of each logo (e.g. Brooklyn's black logo would vanish on a dark background).
  return (
    <picture className="logo">
      <source srcSet={logoUrl(team, true)} media="(prefers-color-scheme: dark)" />
      <img src={logoUrl(team)} alt="" width={size} height={size} style={{ width: size, height: size }} loading="lazy" />
    </picture>
  );
}
