import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiError, mediaUrl } from "../api/client";
import { ChevronRight } from "../components/Icons";
import { Thumb } from "../components/Thumb";
import { JobProgress, UploadDropzone } from "../components/Upload";
import { VideoAsk } from "../components/VideoAsk";
import { useJob, useVideos } from "../hooks/queries";
import { formatTime } from "../lib/format";

const STORAGE_KEY = "spartan.jobs";

interface TrackedJob {
  jobId: string;
  videoId: string;
  title: string;
}

function loadJobs(): TrackedJob[] {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function UploadPage() {
  const [jobs, setJobs] = useState<TrackedJob[]>(loadJobs);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const uploads = useVideos("uploads", 1, 12);
  const client = useQueryClient();

  useEffect(() => {
    document.title = "Upload · Spartan Search";
  }, []);
  useEffect(() => sessionStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.slice(0, 6))), [jobs]);

  async function onFile(file: File) {
    setError(null);
    setBusy(true);
    try {
      const res = await api.upload(file);
      setJobs((j) => [{ jobId: res.job_id, videoId: res.video_id, title: file.name }, ...j]);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const refresh = () => {
    client.invalidateQueries({ queryKey: ["videos"] });
    client.invalidateQueries({ queryKey: ["collections"] });
  };

  return (
    <div className="wrap-app pb-16 pt-12 sm:pt-16">
      <h1 className="h-page">Add your own video.</h1>
      <p className="lead mt-4 max-w-[640px]">It is decoded, split into short windows, encoded by every model, and becomes searchable on its own.</p>

      <div className="mt-10 max-w-[880px]">
        <UploadDropzone onFile={onFile} disabled={busy} />
        {busy && <p className="meta mt-3" role="status">Uploading…</p>}
        {error && (
          <p role="alert" className="mt-3 rounded-card bg-surface px-5 py-3 text-[15px] font-medium shadow-sm">
            {error}
          </p>
        )}
      </div>

      {jobs.length > 0 && (
        <section className="mt-10 max-w-[880px]" aria-labelledby="active">
          <h2 id="active" className="text-[21px] font-semibold tracking-[-0.015em]">
            Processing
          </h2>
          <div className="mt-4 flex flex-col gap-8">
            {jobs.map((j) => (
              <TrackedJobCard key={j.jobId} job={j} onDone={refresh} />
            ))}
          </div>
        </section>
      )}

      <section className="mt-12" aria-labelledby="recent">
        <h2 id="recent" className="text-[21px] font-semibold tracking-[-0.015em]">
          Recent uploads
        </h2>
        {uploads.data && uploads.data.items.length === 0 ? (
          <p className="meta mt-3">Nothing yet. Uploaded videos appear here.</p>
        ) : (
          <ul className="mt-4 grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}>
            {(uploads.data?.items ?? []).map((v) => (
              <li key={v.id} className="card overflow-hidden p-3">
                <Link to={`/video/${v.id}`} aria-label={`Open ${v.title}`}>
                  <Thumb src={mediaUrl(v.thumbnail_url)} />
                </Link>
                <p className="mt-3 truncate font-semibold">{v.title}</p>
                <p className="meta">
                  {formatTime(v.duration)} · <span className="capitalize">{v.status}</span>
                </p>
                {v.status === "ready" && (
                  <Link
                    to={`/video/${v.id}`}
                    className="mt-2 inline-flex items-center gap-0.5 text-[15px] font-medium text-accent-text hover:underline"
                  >
                    Ask about this video <ChevronRight size={14} />
                  </Link>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function TrackedJobCard({ job, onDone }: { job: TrackedJob; onDone: () => void }) {
  const status = useJob(job.jobId);
  const finished = status.data?.status === "indexed";
  useEffect(() => {
    if (finished) onDone();
  }, [finished, onDone]);
  if (!status.data) return <div className="card skeleton h-[150px]" aria-hidden />;
  return (
    <div>
      <JobProgress job={status.data} title={job.title} />
      {finished && (
        <div className="mt-4">
          <VideoAsk videoId={job.videoId} />
        </div>
      )}
    </div>
  );
}
