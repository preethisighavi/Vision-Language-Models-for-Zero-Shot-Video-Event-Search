import { useEffect, useMemo, useState } from "react";

import type { BenchmarkRow } from "../api/client";
import { EmptyState, ErrorState } from "../components/States";
import { useBenchmarks } from "../hooks/queries";
import { formatMetric } from "../lib/format";

const RETRIEVAL = ["R@1", "R@5", "R@10", "MedR", "MRR", "latency_ms"] as const;
const LOCALIZATION = ["R1@0.5", "R1@0.7", "mIoU"] as const;
const LOWER_IS_BETTER = new Set(["MedR", "MeanR", "latency_ms"]);
const HEADERS: Record<string, string> = { latency_ms: "Latency" };

interface Row {
  model: string;
  dataset: string;
  protocol: string;
  notes: string;
  values: Record<string, number>;
}

function pivot(rows: BenchmarkRow[]): Row[] {
  const map = new Map<string, Row>();
  for (const r of rows) {
    const key = `${r.model}|${r.dataset}`;
    const row = map.get(key) ?? { model: r.model, dataset: r.dataset, protocol: r.protocol, notes: r.notes, values: {} };
    row.values[r.metric] = r.value;
    row.protocol ||= r.protocol;
    row.notes ||= r.notes;
    map.set(key, row);
  }
  return [...map.values()];
}

function best(rows: Row[], metric: string): number | undefined {
  const values = rows.map((r) => r.values[metric]).filter((v): v is number => v !== undefined);
  if (!values.length) return undefined;
  return LOWER_IS_BETTER.has(metric) ? Math.min(...values) : Math.max(...values);
}

export function BenchmarksPage() {
  const benchmarks = useBenchmarks();
  const rows = useMemo(() => pivot(benchmarks.data ?? []), [benchmarks.data]);
  const datasets = useMemo(() => [...new Set(rows.map((r) => r.dataset))], [rows]);
  const [dataset, setDataset] = useState<string | null>(null);
  const active = dataset ?? datasets[0];
  const visible = rows.filter((r) => r.dataset === active);
  const hasLocalization = rows.some((r) => LOCALIZATION.some((m) => r.values[m] !== undefined));

  useEffect(() => {
    document.title = "Benchmarks · Spartan Search";
  }, []);

  return (
    <div className="wrap-app pb-16 pt-12 sm:pt-16">
      <h1 className="h-page">Benchmarks</h1>
      <p className="lead mt-4 max-w-[720px]">Retrieval metrics for each model, read from the team's shared results table. The best value in each column is highlighted.</p>

      {benchmarks.error ? (
        <div className="mt-10">
          <ErrorState error={benchmarks.error} onRetry={() => benchmarks.refetch()} />
        </div>
      ) : benchmarks.data && rows.length === 0 ? (
        <div className="mt-10">
          <EmptyState title="No results yet.">Rows appear here once results/model_results.csv has entries.</EmptyState>
        </div>
      ) : (
        <>
          <div role="tablist" aria-label="Dataset" className="mt-8 inline-flex rounded-pill bg-fill p-[3px]">
            {datasets.map((d) => (
              <button
                key={d}
                role="tab"
                type="button"
                aria-selected={d === active}
                onClick={() => setDataset(d)}
                className={`rounded-pill px-4 py-1.5 text-[13px] font-medium transition-all duration-200 ${d === active ? "bg-surface shadow-segment" : "text-text-nav hover:text-text"}`}
              >
                {d}
              </button>
            ))}
          </div>

          <section className="card-lg mt-6 overflow-x-auto p-2 sm:p-4" aria-label={`Retrieval metrics on ${active}`}>
            <table className="w-full min-w-[720px] border-collapse text-left text-[15px]">
              <caption className="sr-only">Retrieval metrics on {active}</caption>
              <thead>
                <tr className="text-[13px] text-text-secondary">
                  <th scope="col" className="px-4 py-3 font-semibold">Model</th>
                  {RETRIEVAL.map((m) => (
                    <th key={m} scope="col" className="px-4 py-3 text-right font-semibold">{HEADERS[m] ?? m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visible.map((r) => (
                  <tr key={r.model} className="border-t border-hairline-soft">
                    <th scope="row" className="px-4 py-3 font-semibold">
                      {r.model}
                      {r.protocol && <span className="block text-[12px] font-normal text-text-secondary">{r.protocol}</span>}
                    </th>
                    {RETRIEVAL.map((m) => {
                      const v = r.values[m];
                      const isBest = v !== undefined && v === best(visible, m) && visible.length > 1;
                      return (
                        <td key={m} className={`px-4 py-3 text-right tabular-nums ${isBest ? "font-semibold text-accent-text" : ""}`}>
                          {v === undefined ? <span className="text-text-secondary">–</span> : formatMetric(m, v)}
                          {isBest && <span className="sr-only"> (best)</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="card-lg mt-6 p-6" aria-labelledby="chart">
            <h2 id="chart" className="text-[19px] font-semibold tracking-[-0.015em]">R@1 by model · {active}</h2>
            <ul className="mt-5 flex flex-col gap-3">
              {visible
                .filter((r) => r.values["R@1"] !== undefined)
                .sort((x, y) => y.values["R@1"] - x.values["R@1"])
                .map((r) => (
                  <li key={r.model} className="grid grid-cols-[minmax(120px,200px)_1fr_64px] items-center gap-3 text-[14px]">
                    <span className="truncate">{r.model}</span>
                    <span className="h-3 rounded-pill bg-fill">
                      <span className="block h-3 rounded-pill bg-accent" style={{ width: `${r.values["R@1"] * 100}%` }} />
                    </span>
                    <span className="text-right font-semibold tabular-nums">{formatMetric("R@1", r.values["R@1"])}</span>
                  </li>
                ))}
            </ul>
          </section>

          <section className="card-lg mt-6 p-6" aria-labelledby="loc">
            <h2 id="loc" className="text-[19px] font-semibold tracking-[-0.015em]">Moment localization</h2>
            {hasLocalization ? (
              <table className="mt-4 w-full text-left text-[15px]">
                <thead>
                  <tr className="text-[13px] text-text-secondary">
                    <th scope="col" className="py-2 font-semibold">Model</th>
                    {LOCALIZATION.map((m) => (
                      <th key={m} scope="col" className="py-2 text-right font-semibold">{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows
                    .filter((r) => LOCALIZATION.some((m) => r.values[m] !== undefined))
                    .map((r) => (
                      <tr key={`${r.model}-${r.dataset}`} className="border-t border-hairline-soft">
                        <th scope="row" className="py-2 font-semibold">{r.model} <span className="font-normal text-text-secondary">· {r.dataset}</span></th>
                        {LOCALIZATION.map((m) => (
                          <td key={m} className="py-2 text-right tabular-nums">{r.values[m] === undefined ? "–" : formatMetric(m, r.values[m])}</td>
                        ))}
                      </tr>
                    ))}
                </tbody>
              </table>
            ) : (
              <p className="meta mt-2">[pending] No localization results yet. R1@0.5, R1@0.7 and mIoU need a dataset with event timestamps.</p>
            )}
          </section>

          {visible.some((r) => r.notes) && (
            <ul className="meta mt-6 flex flex-col gap-1">
              {visible.filter((r) => r.notes).map((r) => (
                <li key={r.model}>
                  <span className="font-semibold text-text">{r.model}:</span> {r.notes}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
