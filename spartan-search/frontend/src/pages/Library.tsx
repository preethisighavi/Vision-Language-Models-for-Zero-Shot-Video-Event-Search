import { useEffect } from "react";
import { Link } from "react-router-dom";

import { mediaUrl } from "../api/client";
import { EmptyState, ErrorState } from "../components/States";
import { Thumb } from "../components/Thumb";
import { useCollections, useVideos } from "../hooks/queries";
import { useUrlState } from "../hooks/useSearchState";
import { formatTime, plural } from "../lib/format";

const PAGE_SIZE = 24;

export function LibraryPage() {
  const { params, update } = useUrlState();
  const collections = useCollections();
  const active = params.get("collection") ?? collections.data?.find((c) => c.video_count > 0)?.id ?? collections.data?.[0]?.id;
  const page = Math.max(1, Number(params.get("page") ?? 1));
  const videos = useVideos(active, page, PAGE_SIZE);
  const pages = videos.data ? Math.max(1, Math.ceil(videos.data.total / PAGE_SIZE)) : 1;

  useEffect(() => {
    document.title = "Library · Spartan Search";
  }, []);

  return (
    <div className="wrap-app pb-16 pt-12 sm:pt-16">
      <h1 className="h-page">Library</h1>
      <p className="lead mt-4">Every clip you can search, by collection.</p>

      <div role="tablist" aria-label="Collections" className="mt-8 inline-flex max-w-full flex-wrap rounded-pill bg-fill p-[3px]">
        {(collections.data ?? []).map((c) => (
          <button
            key={c.id}
            role="tab"
            type="button"
            aria-selected={c.id === active}
            onClick={() => update({ collection: c.id, page: null })}
            className={`rounded-pill px-4 py-1.5 text-[13px] font-medium transition-all duration-200 ease-out ${
              c.id === active ? "bg-surface text-text shadow-segment" : "text-text-nav hover:text-text"
            }`}
          >
            {c.name} <span className="text-text-secondary">{c.video_count}</span>
          </button>
        ))}
      </div>

      <div className="mt-8" role="tabpanel" aria-label={collections.data?.find((c) => c.id === active)?.name}>
        {videos.error ? (
          <ErrorState error={videos.error} onRetry={() => videos.refetch()} />
        ) : videos.data && videos.data.items.length === 0 ? (
          <EmptyState title="This library is empty." action={active === "uploads" ? <Link className="btn-primary" to="/upload">Upload a video</Link> : undefined}>
            Add clips with the seed script or the admin import endpoint.
          </EmptyState>
        ) : (
          <>
            <p className="meta">{videos.data ? plural(videos.data.total, "video") : " "}</p>
            <ul className="mt-4 grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
              {(videos.data?.items ?? Array.from({ length: 8 }, () => null)).map((v, i) =>
                v ? (
                  <li key={v.id}>
                    <Link to={`/video/${v.id}`} className="group block rounded-card">
                      <Thumb src={mediaUrl(v.thumbnail_url)} className="shadow-sm transition-shadow duration-200 ease-out group-hover:shadow-md">
                        <span className="absolute bottom-1.5 right-1.5 rounded-[8px] bg-black/70 px-1.5 py-0.5 text-[12px] font-medium tabular-nums text-white">
                          {formatTime(v.duration)}
                        </span>
                      </Thumb>
                      <p className="mt-2 line-clamp-2 text-[15px] font-semibold leading-snug">{v.title}</p>
                      {v.status !== "ready" && <p className="meta capitalize">{v.status}</p>}
                    </Link>
                  </li>
                ) : (
                  <li key={i} aria-hidden>
                    <div className="skeleton aspect-video rounded-thumb" />
                    <div className="skeleton mt-2 h-4 w-3/4 rounded" />
                  </li>
                ),
              )}
            </ul>
            {pages > 1 && (
              <nav aria-label="Pages" className="mt-8 flex items-center justify-center gap-3">
                <button type="button" className="btn-secondary" disabled={page <= 1} onClick={() => update({ page: String(page - 1) })}>
                  Previous
                </button>
                <span className="meta tabular-nums">
                  Page {page} of {pages}
                </span>
                <button type="button" className="btn-secondary" disabled={page >= pages} onClick={() => update({ page: String(page + 1) })}>
                  Next
                </button>
              </nav>
            )}
          </>
        )}
      </div>
    </div>
  );
}
