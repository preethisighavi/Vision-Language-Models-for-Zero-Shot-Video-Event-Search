import { useEffect } from "react";
import { Link, useParams } from "react-router-dom";

import { VideoAsk } from "../components/VideoAsk";
import { ErrorState } from "../components/States";
import { useCollections, useVideo } from "../hooks/queries";
import { useUrlState } from "../hooks/useSearchState";
import { formatSpan, formatTime } from "../lib/format";

export function VideoDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { params } = useUrlState();
  const video = useVideo(id);
  const collections = useCollections();
  const collection = collections.data?.find((c) => c.id === video.data?.collection);

  useEffect(() => {
    if (video.data) document.title = `${video.data.title} · Spartan Search`;
  }, [video.data]);

  if (video.error) {
    return (
      <div className="wrap-app py-12">
        <ErrorState error={video.error} />
      </div>
    );
  }
  const v = video.data;

  return (
    <div className="wrap-app pb-16 pt-8 sm:pt-10">
      <p className="meta">
        <Link to={`/library?collection=${v?.collection ?? ""}`} className="link">
          {collection?.name ?? "Library"}
        </Link>
      </p>
      <h1 className="h-results mt-1">{v?.title ?? " "}</h1>
      <p className="lead mt-3 max-w-[720px]">Ask a question. The model answers with the moment in this video.</p>

      <div className="mt-8">{id && <VideoAsk videoId={id} initialModel={params.get("model") ?? undefined} />}</div>

      {v && (
        <section className="mt-10 grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))" }} aria-label="Details">
          <div className="card p-6">
            <h2 className="text-[19px] font-semibold tracking-[-0.015em]">About this video</h2>
            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-[15px]">
              <dt className="text-text-secondary">Duration</dt>
              <dd>{formatTime(v.duration)}</dd>
              <dt className="text-text-secondary">Resolution</dt>
              <dd>{v.width && v.height ? `${v.width} × ${v.height}` : "–"}</dd>
              <dt className="text-text-secondary">Source</dt>
              <dd>{v.source || "–"}</dd>
              <dt className="text-text-secondary">Licence</dt>
              <dd>{v.license || "–"}</dd>
              <dt className="text-text-secondary">Status</dt>
              <dd className="capitalize">{v.status}</dd>
            </dl>
          </div>
          <div className="card p-6">
            <h2 className="text-[19px] font-semibold tracking-[-0.015em]">Indexed windows</h2>
            <p className="meta mt-1">The video is split into these windows before a model can answer.</p>
            <ul className="mt-3 flex flex-wrap gap-2">
              {v.windows.map((w) => (
                <li key={w.id} className="rounded-pill bg-bg px-3 py-1 text-[13px] tabular-nums text-text-nav">
                  {formatSpan(w)}
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
