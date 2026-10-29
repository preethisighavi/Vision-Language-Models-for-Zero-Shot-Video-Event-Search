import { mediaUrl, type SearchResult } from "../api/client";
import { formatMatch, formatSpan } from "../lib/format";
import { Thumb } from "./Thumb";

export function RankChange({ change }: { change: number | "new" | null }) {
  if (change === null) return <span className="w-10 text-right text-[13px] text-text-secondary">–</span>;
  if (change === "new") return <span className="w-10 text-right text-[13px] text-text-secondary">new</span>;
  if (change === 0) return <span className="w-10 text-right text-[13px] text-text-secondary" aria-label="Same rank">–</span>;
  return change > 0 ? (
    <span className="w-10 text-right text-[13px] font-semibold text-accent-text" aria-label={`Up ${change}`}>
      ↑ {change}
    </span>
  ) : (
    <span className="w-10 text-right text-[13px] text-text-secondary" aria-label={`Down ${-change}`}>
      ↓ {-change}
    </span>
  );
}

export function CompareColumn({
  name,
  description,
  metricLabel,
  results,
  baseline,
  groundTruth,
  latencyMs,
}: {
  name: string;
  description: string;
  metricLabel: string;
  results: SearchResult[];
  baseline?: SearchResult[];
  groundTruth?: string | null;
  latencyMs?: number;
}) {
  const baseRank = new Map((baseline ?? []).map((r) => [r.video_id, r.rank]));
  return (
    <section className="card-lg flex flex-col p-5 sm:p-6" aria-label={name}>
      <header className="flex items-start justify-between gap-4 border-b border-hairline-soft pb-4">
        <div className="min-w-0">
          <h2 className="h-card">{name}</h2>
          <p className="meta mt-1">{description}</p>
        </div>
        <div className="shrink-0 text-right">
          <p className="text-[24px] font-semibold tracking-[-0.02em]">{metricLabel}</p>
          <p className="text-[12px] text-text-secondary">R@1 benchmark{latencyMs !== undefined ? ` · ${latencyMs} ms` : ""}</p>
        </div>
      </header>
      <ol className="mt-2 flex flex-col">
        {results.map((r) => {
          const prev = baseRank.get(r.video_id);
          const change = !baseline ? null : prev === undefined ? ("new" as const) : prev - r.rank;
          const correct = groundTruth && r.video_id === groundTruth;
          return (
            <li key={r.video_id} className="flex items-center gap-3 border-b border-hairline-soft py-3 last:border-0">
              <span className="w-5 text-[15px] font-semibold tabular-nums text-text-secondary">{r.rank}</span>
              <Thumb src={mediaUrl(r.thumbnail_url)} className="w-[88px] shrink-0" rounded="rounded-thumb-sm" showPlay={false} />
              <div className="min-w-0 flex-1">
                <p className="truncate text-[15px] font-semibold">{r.title}</p>
                <p className="text-[13px] text-text-secondary">
                  <span className="tabular-nums">{r.spans[0] ? formatSpan(r.spans[0]) : "whole clip"}</span> ·{" "}
                  <span className="text-accent-text">{formatMatch(r.score)}</span>
                </p>
              </div>
              {correct && (
                <span className="rounded-pill bg-accent-tint px-2.5 py-0.5 text-[12px] font-semibold text-accent-text">Correct</span>
              )}
              <RankChange change={change} />
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export function MetricTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="card p-5">
      <p className="meta">{label}</p>
      <p className="mt-1 text-[32px] font-semibold tracking-[-0.025em]">{value}</p>
      {hint && <p className="mt-1 text-[13px] text-text-secondary">{hint}</p>}
    </div>
  );
}
