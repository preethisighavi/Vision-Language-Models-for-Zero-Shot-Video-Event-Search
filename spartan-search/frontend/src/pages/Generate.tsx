import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiError, mediaUrl, streamUrl } from "../api/client";
import { ChevronRight } from "../components/Icons";
import { Thumb } from "../components/Thumb";
import { JobProgress } from "../components/Upload";
import { useJob, useVideos } from "../hooks/queries";
import { formatTime } from "../lib/format";

const STORAGE_KEY = "spartan.genjobs";
const EXAMPLES = [
  "a man is playing a guitar on stage",
  "a cat is playing with a ball of yarn",
  "a car is driving down a city street at night",
  "a woman is cooking pasta in a kitchen",
];

interface TrackedGeneration {
  jobId: string;
  videoId: string;
  prompt: string;
  model: string;
}

function loadJobs(): TrackedGeneration[] {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function GeneratePage() {
  const status = useQuery({ queryKey: ["generator"], queryFn: api.generatorStatus, staleTime: 30_000 });
  const models = status.data?.models ?? [];
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const [seed, setSeed] = useState(0);
  const [jobs, setJobs] = useState<TrackedGeneration[]>(loadJobs);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const generated = useVideos("generated", 1, 12);
  const client = useQueryClient();
  const promptId = useId();
  const seedId = useId();

  useEffect(() => {
    document.title = "Generate · Spartan Search";
  }, []);
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.slice(0, 6)));
    } catch {
      /* storage unavailable: tracking is per page view only */
    }
  }, [jobs]);

  const selected = model ?? models[0]?.id ?? "";
  const available = status.data?.available ?? false;
  const canSubmit = available && !busy && prompt.trim().length >= 3 && !!selected;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setBusy(true);
    try {
      const res = await api.generate({ prompt: prompt.trim(), model: selected, seed });
      const label = models.find((m) => m.id === selected)?.label ?? selected;
      setJobs((j) => [{ jobId: res.job_id, videoId: res.video_id, prompt: prompt.trim(), model: label }, ...j]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Generation failed. Try again.");
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
      <h1 className="h-page">Make a video from text.</h1>
      <p className="lead mt-4 max-w-[680px]">
        Describe a short scene. Our model, trained from scratch on MSR-VTT and MSVD, makes a 5-second 128px clip at 3 frames per second; the
        pretrained Wan 2.1 baseline makes a longer 480p one. Every clip is indexed, so you can search it right after.
      </p>

      {status.isSuccess && !available && (
        <p role="status" className="mt-8 max-w-[880px] rounded-card bg-surface px-5 py-3 text-[15px] font-medium shadow-sm">
          The generation server is offline right now. Recent generations are still shown below.
        </p>
      )}

      <form onSubmit={submit} className="card-lg mt-10 max-w-[880px] p-5 sm:p-6">
        <label htmlFor={promptId} className="font-semibold">
          Prompt
        </label>
        <textarea
          id={promptId}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          maxLength={300}
          rows={3}
          placeholder="a dog is running on the beach"
          className="mt-2 w-full resize-y rounded-[12px] border border-hairline bg-bg px-4 py-3 text-[17px] outline-none focus-visible:border-accent"
        />
        <div className="mt-2 flex flex-wrap gap-2" aria-label="Example prompts">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" className="btn-secondary !min-h-[32px] !px-3 !text-[13px]" onClick={() => setPrompt(ex)}>
              {ex}
            </button>
          ))}
        </div>

        <fieldset className="mt-5">
          <legend className="font-semibold">Model</legend>
          <div className="mt-2 flex flex-wrap gap-2">
            {models.length === 0 && <span className="meta">{status.isLoading ? "Loading models…" : "No models available"}</span>}
            {models.map((m) => (
              <label
                key={m.id}
                className={`cursor-pointer rounded-pill px-4 py-2 text-[15px] font-medium transition-colors ${
                  selected === m.id ? "bg-accent text-white" : "bg-fill text-text hover:bg-track"
                }`}
              >
                <input type="radio" name="model" value={m.id} checked={selected === m.id} onChange={() => setModel(m.id)} className="sr-only" />
                {m.label}
              </label>
            ))}
          </div>
        </fieldset>

        <div className="mt-5 flex flex-wrap items-end justify-between gap-4">
          <div>
            <label htmlFor={seedId} className="font-semibold">
              Seed
            </label>
            <input
              id={seedId}
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
              className="mt-2 block w-[120px] rounded-[12px] border border-hairline bg-bg px-3 py-2 outline-none focus-visible:border-accent"
            />
          </div>
          <button type="submit" className="btn-primary" disabled={!canSubmit}>
            {busy ? "Sending…" : "Generate"}
          </button>
        </div>
        {error && (
          <p role="alert" className="mt-4 text-[15px] font-medium">
            {error}
          </p>
        )}
      </form>

      {jobs.length > 0 && (
        <section className="mt-10 max-w-[880px]" aria-labelledby="gen-active">
          <h2 id="gen-active" className="text-[21px] font-semibold tracking-[-0.015em]">
            Your generations
          </h2>
          <div className="mt-4 flex flex-col gap-8">
            {jobs.map((j) => (
              <GenerationCard key={j.jobId} job={j} onDone={refresh} />
            ))}
          </div>
        </section>
      )}

      <section className="mt-12" aria-labelledby="gen-recent">
        <h2 id="gen-recent" className="text-[21px] font-semibold tracking-[-0.015em]">
          Recent generations
        </h2>
        {generated.data && generated.data.items.length === 0 ? (
          <p className="meta mt-3">Nothing yet. Generated clips appear here.</p>
        ) : (
          <ul className="mt-4 grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}>
            {(generated.data?.items ?? []).map((v) => (
              <li key={v.id} className="card overflow-hidden p-3">
                <Link to={`/video/${v.id}`} aria-label={`Open ${v.title}`}>
                  <Thumb src={mediaUrl(v.thumbnail_url)} />
                </Link>
                <p className="mt-3 truncate font-semibold">{v.title}</p>
                <p className="meta">
                  {formatTime(v.duration)} · <span className="capitalize">{v.status}</span>
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function GenerationCard({ job, onDone }: { job: TrackedGeneration; onDone: () => void }) {
  const status = useJob(job.jobId);
  const finished = status.data?.status === "indexed";
  useEffect(() => {
    if (finished) onDone();
  }, [finished, onDone]);
  if (!status.data) return <div className="card skeleton h-[150px]" aria-hidden />;
  return (
    <div>
      <JobProgress job={status.data} title={`${job.prompt} · ${job.model}`} kind="generate" />
      {finished && (
        <div className="card mt-4 p-3">
          <video
            src={streamUrl(job.videoId)}
            controls
            loop
            autoPlay
            muted
            playsInline
            className="aspect-video w-full rounded-[12px] bg-black object-contain"
            style={{ imageRendering: "auto" }}
          />
          <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
            <Link
              to={`/search?q=${encodeURIComponent(job.prompt)}&collection=generated`}
              className="inline-flex items-center gap-0.5 text-[15px] font-medium text-accent-text hover:underline"
            >
              Search for it <ChevronRight size={14} />
            </Link>
            <Link to={`/video/${job.videoId}`} className="inline-flex items-center gap-0.5 text-[15px] font-medium text-accent-text hover:underline">
              Open video page <ChevronRight size={14} />
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
