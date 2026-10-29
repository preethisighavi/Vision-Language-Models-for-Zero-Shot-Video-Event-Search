import { useEffect, useMemo, useRef } from "react";

import type { BenchmarkRow, ModelInfo } from "../api/client";
import { CompareColumn } from "../components/Compare";
import { LibrarySelect, ModelSwitch, SearchBar } from "../components/Controls";
import { EmptyState, ErrorState, IndexBuildingBanner, isIndexNotReady } from "../components/States";
import { useBenchmarks, useCollections, useCompare, useModels, useSlashFocus } from "../hooks/queries";
import { resolveSelection, useUrlState } from "../hooks/useSearchState";
import { formatMetric } from "../lib/format";

const PREFERRED_A = ["clip-b32-ft"];
const PREFERRED_B = ["spartan-v1"];

function pickPair(models: ModelInfo[], a: string | null, b: string | null): [string, string] {
  const ids = models.map((m) => m.id);
  const first = (a && ids.includes(a) ? a : PREFERRED_A.find((x) => ids.includes(x))) ?? ids.find((x) => x !== b) ?? "";
  const second = (b && ids.includes(b) && b !== first ? b : PREFERRED_B.find((x) => ids.includes(x) && x !== first)) ?? ids.find((x) => x !== first) ?? "";
  return [first, second];
}

/** Which /benchmarks row belongs to which registered model. Add an entry when a model gets results. */
const BENCHMARK_NAMES: Record<string, string> = {
  "clip-b32": "CLIP (zero-shot)",
  "clip-b32-ft": "CLIP (fine-tuned)",
  "spartan-v1": "Spartan (v1)",
};

/** R@1 for a model on the dataset behind the selected library, from /benchmarks. */
function headlineMetric(rows: BenchmarkRow[] | undefined, model: ModelInfo | undefined, collection: string): string {
  if (!rows || !model) return "[pending]";
  const dataset = collection.startsWith("msrvtt") ? "MSR-VTT" : collection.startsWith("msvd") ? "MSVD" : "";
  const name = BENCHMARK_NAMES[model.id] ?? model.name;
  const row = rows.find((r) => r.metric === "R@1" && r.dataset === dataset && r.model === name);
  return row ? formatMetric("R@1", row.value) : "[pending]";
}

export function ComparePage() {
  const { params, update } = useUrlState();
  const models = useModels();
  const collections = useCollections();
  const benchmarks = useBenchmarks();
  const inputRef = useRef<HTMLInputElement>(null);
  useSlashFocus(inputRef);

  const q = params.get("q") ?? "";
  const { collection } = resolveSelection(params, models.data, collections.data);
  const [a, b] = models.data ? pickPair(models.data, params.get("a"), params.get("b")) : ["", ""];
  const req = useMemo(() => (q && a && b && collection ? { query: q, models: [a, b], collection, top_k: 5 } : null), [q, a, b, collection]);
  const compare = useCompare(req);
  const info = (id: string) => models.data?.find((m) => m.id === id);

  useEffect(() => {
    document.title = "Compare · Spartan Search";
  }, []);

  return (
    <div className="wrap-app pb-16 pt-12 sm:pt-16">
      <h1 className="h-page">Same question. Two models.</h1>
      <p className="lead mt-4 max-w-[640px]">Run one sentence through two models and see where they agree, and where the ranking moves.</p>

      <div className="mt-8 flex flex-col gap-3">
        <SearchBar ref={inputRef} initial={q} onSubmit={(v) => update({ q: v, a, b, collection })} />
        <div className="flex flex-wrap items-center gap-3">
          {models.data && models.data.length >= 2 && (
            <>
              <span className="meta">First</span>
              <ModelSwitch label="First model" models={models.data} value={a} onChange={(v) => update({ a: v, b: v === b ? a : b })} />
              <span className="meta">Second</span>
              <ModelSwitch label="Second model" models={models.data} value={b} onChange={(v) => update({ b: v, a: v === a ? b : a })} />
            </>
          )}
          {collections.data && <LibrarySelect collections={collections.data} value={collection} onChange={(v) => update({ collection: v })} />}
        </div>
      </div>

      <div className="mt-10">
        {models.data && models.data.length < 2 ? (
          <EmptyState title="Compare needs two models.">Enable a second model in models.yaml to compare rankings.</EmptyState>
        ) : !q ? (
          <EmptyState title="Ask the same question twice.">Type a sentence above. Each model returns its top five, side by side.</EmptyState>
        ) : compare.error ? (
          isIndexNotReady(compare.error) ? <IndexBuildingBanner message={compare.error.message} /> : <ErrorState error={compare.error} onRetry={() => compare.refetch()} />
        ) : (
          <>
            <div className="grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))" }} aria-busy={compare.isFetching}>
              {(compare.data?.runs ?? []).map((run, i) => (
                <CompareColumn
                  key={run.model}
                  name={info(run.model)?.name ?? run.model}
                  description={info(run.model)?.description ?? ""}
                  metricLabel={headlineMetric(benchmarks.data, info(run.model), collection)}
                  results={run.results}
                  baseline={i > 0 ? compare.data!.runs[0].results : undefined}
                  groundTruth={compare.data!.ground_truth_video_id}
                  latencyMs={run.latency_ms}
                />
              ))}
              {compare.isPending &&
                [0, 1].map((i) => <div key={i} className="card-lg skeleton h-[420px]" aria-hidden />)}
            </div>
            <p className="meta mt-6 max-w-[760px]">
              In MSR-VTT 1k-A each query has one correct video. Arrows show the change in rank versus the first model.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
