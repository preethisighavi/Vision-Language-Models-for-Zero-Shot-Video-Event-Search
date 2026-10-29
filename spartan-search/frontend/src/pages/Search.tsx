import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";

import { streamUrl } from "../api/client";
import { LibrarySelect, ModelSwitch, RerankSwitch, SearchBar } from "../components/Controls";
import { Player } from "../components/Player";
import { ResultCard } from "../components/ResultCard";
import { EmptyState, ErrorState, IndexBuildingBanner, isIndexNotReady, PlayerSkeleton, ResultSkeleton } from "../components/States";
import { useCollections, useModels, useSearch, useSlashFocus, useVideo } from "../hooks/queries";
import { resolveSelection, useUrlState } from "../hooks/useSearchState";
import { formatTime } from "../lib/format";
import { EXAMPLES } from "./Home";

export function SearchPage() {
  const { params, update } = useUrlState();
  const models = useModels();
  const collections = useCollections();
  const inputRef = useRef<HTMLInputElement>(null);
  const playerRef = useRef<HTMLDivElement>(null);
  useSlashFocus(inputRef);

  const q = params.get("q") ?? "";
  const videoFilter = params.get("video") ?? undefined;
  const scoped = useVideo(videoFilter);
  const { model, collection } = resolveSelection(params, models.data, collections.data);
  const modelInfo = models.data?.find((m) => m.id === model);
  const collectionInfo = collections.data?.find((c) => c.id === collection);
  const rerank = params.get("rerank") !== "false";

  const search = useSearch(q && model && collection ? { query: q, model, collection, rerank, videoId: videoFilter } : null);
  const results = search.data?.results ?? [];
  const selectedId = params.get("selected") ?? results[0]?.video_id;
  const selected = useMemo(() => results.find((r) => r.video_id === selectedId) ?? results[0], [results, selectedId]);

  useEffect(() => {
    document.title = q ? `“${q}” · Spartan Search` : "Search · Spartan Search";
  }, [q]);

  const select = (id: string) => {
    update({ selected: id }, true);
    if (window.matchMedia("(max-width: 959px)").matches) {
      requestAnimationFrame(() => playerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  };

  return (
    <div className="wrap-app pb-16 pt-8 sm:pt-10">
      <div className="flex flex-col gap-3">
        <SearchBar ref={inputRef} initial={q} onSubmit={(v) => update({ q: v, selected: null, model, collection })} />
        <div className="flex flex-wrap items-center gap-3">
          {models.data && <ModelSwitch models={models.data} value={model} onChange={(v) => update({ model: v, selected: null })} />}
          {collections.data && (
            <LibrarySelect collections={collections.data} value={collection} onChange={(v) => update({ collection: v, selected: null, video: null })} />
          )}
          {modelInfo?.supports_rerank && <RerankSwitch checked={rerank} onChange={(v) => update({ rerank: v ? null : "false", selected: null })} />}
          {videoFilter && (
            <button type="button" className="btn-secondary h-[34px] min-h-0 px-4 text-[13px]" onClick={() => update({ video: null, selected: null })}>
              Inside {scoped.data?.title ?? "this video"} · Search the whole library
            </button>
          )}
        </div>
      </div>

      {!q ? (
        <div className="mt-12">
          <EmptyState title={videoFilter ? `Search inside ${scoped.data?.title ?? "this video"}.` : "What are you looking for?"} action={
            <ul className="mt-2 flex flex-wrap justify-center gap-2">
              {EXAMPLES.map((ex) => (
                <li key={ex}>
                  <button type="button" className="btn-secondary h-9 min-h-0 text-[14px]" onClick={() => update({ q: ex, model, collection })}>
                    {ex}
                  </button>
                </li>
              ))}
            </ul>
          }>
            {videoFilter
              ? "Pick a model above, then describe a moment. The search stays inside this video."
              : "Describe a moment in a sentence, then press Enter."}
          </EmptyState>
        </div>
      ) : (
        <>
          <header className="mt-10" aria-live="polite">
            <h1 className="h-results">Results for “{q}”</h1>
            <p className="meta mt-1.5">
              {search.data
                ? videoFilter
                  ? `Inside ${scoped.data?.title ?? "this video"} · ${modelInfo?.name ?? model} · ${search.data.latency_ms.total} ms`
                  : `Top ${results.length} of ${search.data.total_videos.toLocaleString()} videos · ${modelInfo?.name ?? model} · ${search.data.latency_ms.total} ms`
                : search.isFetching
                  ? "Searching…"
                  : " "}
            </p>
          </header>

          {search.error ? (
            <div className="mt-6">
              {isIndexNotReady(search.error) ? (
                <IndexBuildingBanner message={search.error.message} />
              ) : (
                <ErrorState error={search.error} onRetry={() => search.refetch()} />
              )}
            </div>
          ) : (
            <div className="mt-6 flex flex-wrap items-start gap-6">
              <ol className="flex min-w-0 flex-col gap-3" style={{ flex: "999 1 520px" }} aria-label="Search results" aria-busy={search.isFetching}>
                {search.isPending
                  ? Array.from({ length: 5 }, (_, i) => <ResultSkeleton key={i} />)
                  : results.map((r) => (
                      <ResultCard
                        key={r.video_id}
                        result={r}
                        selected={r.video_id === selected?.video_id}
                        collectionName={collectionInfo?.name}
                        onSelect={() => select(r.video_id)}
                      />
                    ))}
                {!search.isPending && results.length === 0 && (
                  <li>
                    <EmptyState title="No matches yet.">
                      Try fewer words, or another library. {collectionInfo?.video_count === 0 && <Link className="link" to="/upload">Upload a video</Link>}
                    </EmptyState>
                  </li>
                )}
              </ol>

              <div ref={playerRef} className="min-w-0 scroll-mt-20 lg:sticky lg:top-[72px]" style={{ flex: "1 1 440px" }}>
                {search.isPending ? (
                  <PlayerSkeleton />
                ) : selected ? (
                  <>
                    <Player
                      videoId={selected.video_id}
                      title={selected.title}
                      streamUrl={streamUrl(selected.video_id)}
                      duration={selected.duration}
                      score={selected.score}
                      meta={`${collectionInfo?.name ?? selected.collection} · ${formatTime(selected.duration)}`}
                      spans={selected.spans}
                      saliency={selected.saliency}
                      modelName={modelInfo?.name}
                      queryId={search.data?.query_id}
                    />
                    <p className="mt-3 px-2 text-[13px]">
                      <Link className="link" to={`/video/${selected.video_id}?${new URLSearchParams({ q, model })}`}>
                        Open video details
                      </Link>
                    </p>
                  </>
                ) : null}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
