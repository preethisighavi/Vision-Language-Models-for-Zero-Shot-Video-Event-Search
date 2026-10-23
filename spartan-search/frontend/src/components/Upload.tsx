import { useId, useState } from "react";

import type { JobInfo } from "../api/client";
import { CheckIcon, UploadIcon } from "./Icons";

export function UploadDropzone({
  onFile,
  disabled,
  maxMb = 200,
  maxMinutes = 10,
}: {
  onFile: (file: File) => void;
  disabled?: boolean;
  maxMb?: number;
  maxMinutes?: number;
}) {
  const [over, setOver] = useState(false);
  const id = useId();
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file && !disabled) onFile(file);
      }}
      className={`card-lg relative transition-colors duration-200 ease-out ${over ? "bg-accent-tint" : ""}`}
    >
      <div
        className={`m-[14px] flex flex-col items-center gap-3 rounded-[20px] border-2 border-dashed px-6 py-14 text-center transition-colors duration-200 ${
          over ? "border-accent" : "border-track"
        }`}
      >
        <span className="grid h-14 w-14 place-items-center rounded-full bg-accent-tint text-accent" aria-hidden>
          <UploadIcon size={26} />
        </span>
        <p className="text-[21px] font-semibold tracking-[-0.015em]">Drop a video here</p>
        <p className="meta">
          MP4, MOV or WebM · up to {maxMb} MB · up to {maxMinutes} minutes
        </p>
        <input
          id={id}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
          className="peer sr-only"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
        <label
          htmlFor={id}
          className={`btn-primary mt-1 cursor-pointer peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent ${disabled ? "pointer-events-none opacity-50" : ""}`}
        >
          Choose a file
        </label>
      </div>
    </div>
  );
}

const STEPS: { key: string; label: string }[] = [
  { key: "uploaded", label: "Uploaded" },
  { key: "decoding", label: "Decoded" },
  { key: "embedding", label: "Embedding" },
  { key: "indexed", label: "Indexed" },
];

function stepIndex(status: JobInfo["status"]): number {
  switch (status) {
    case "queued":
      return 0;
    case "uploaded":
      return 1;
    case "decoding":
      return 1;
    case "embedding":
      return 2;
    case "indexed":
      return 4;
    default:
      return 0;
  }
}

/** Four-step stepper: done steps are filled, the current one spins, pending ones are grey rings. */
export function JobProgress({ job, title }: { job: JobInfo; title: string }) {
  const failed = job.status === "failed";
  const current = stepIndex(job.status);
  const pct = Math.round(Math.min(Math.max(job.progress, 0), 1) * 100);
  return (
    <div className="card p-5">
      <p className="truncate font-semibold">{title}</p>
      <ol className="mt-4 flex items-start justify-between gap-2" aria-label="Upload progress">
        {STEPS.map((s, i) => {
          const done = i < current;
          const active = i === current && !failed;
          return (
            <li key={s.key} className="flex flex-1 flex-col items-center gap-1.5 text-center" aria-current={active ? "step" : undefined}>
              <span className="relative grid h-7 w-7 place-items-center">
                {done ? (
                  <span className="grid h-7 w-7 place-items-center rounded-full bg-accent text-white">
                    <CheckIcon />
                  </span>
                ) : active ? (
                  <span className="h-7 w-7 animate-spin rounded-full border-[3px] border-fill border-t-accent" />
                ) : (
                  <span className={`h-7 w-7 rounded-full border-[3px] ${failed && i === current ? "border-text-secondary" : "border-track"}`} />
                )}
              </span>
              <span className={`text-[12px] font-medium ${done || active ? "text-text" : "text-text-secondary"}`}>{s.label}</span>
            </li>
          );
        })}
      </ol>
      <div
        className="mt-4 h-1.5 w-full overflow-hidden rounded-pill bg-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label="Upload progress"
      >
        <div className="h-full rounded-pill bg-accent transition-[width] duration-300 ease-out" style={{ width: `${pct}%` }} />
      </div>
      <p className={`mt-2 text-[13px] ${failed ? "font-medium text-text" : "text-text-secondary"}`} role="status">
        {failed ? `Failed: ${job.error ?? "processing error"}` : `${job.step_detail || "Working"} · ${pct}%`}
      </p>
    </div>
  );
}
