import { useCallback, useEffect, useRef, useState } from "react";

import type { Span } from "../api/client";
import { useFeedback } from "../hooks/queries";
import { formatMatch, formatSpan, formatTime, matchPercent } from "../lib/format";
import { Timeline } from "./Timeline";

export interface PlayerProps {
  videoId: string;
  title: string;
  streamUrl: string;
  duration: number;
  score?: number | null;
  meta?: string;
  spans: Span[];
  saliency?: number[] | null;
  modelName?: string;
  queryId?: string;
  showFeedback?: boolean;
}

/** Seeks to the top span and loops it until the user scrubs; moment chips switch the looped span. */
export function Player({ videoId, title, streamUrl, duration, score, meta, spans, saliency, modelName, queryId, showFeedback = true }: PlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [active, setActive] = useState(0);
  const [looping, setLooping] = useState(true);
  const [time, setTime] = useState(0);
  const programmatic = useRef(false);
  const span = spans[active];

  const seekTo = useCallback((t: number) => {
    const v = videoRef.current;
    if (!v) return;
    programmatic.current = true;
    v.currentTime = t;
  }, []);

  const playSpan = useCallback(
    (i: number) => {
      setActive(i);
      setLooping(true);
      const s = spans[i];
      if (s) seekTo(s.start);
      videoRef.current?.play().catch(() => undefined);
    },
    [spans, seekTo],
  );

  useEffect(() => {
    setActive(0);
    setLooping(true);
  }, [videoId, spans]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const start = () => {
      if (spans[0]) seekTo(spans[0].start);
      v.play().catch(() => undefined);
    };
    if (v.readyState >= 1) start();
    else v.addEventListener("loadedmetadata", start, { once: true });
    return () => v.removeEventListener("loadedmetadata", start);
  }, [videoId, streamUrl, spans, seekTo]);

  function onTimeUpdate() {
    const v = videoRef.current;
    if (!v) return;
    setTime(v.currentTime);
    if (looping && span && v.currentTime >= span.end) seekTo(span.start);
  }

  function onSeeking() {
    if (programmatic.current) {
      programmatic.current = false;
      return;
    }
    setLooping(false); // the user scrubbed: stop looping
  }

  function onChipKey(e: React.KeyboardEvent, i: number) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const next = (i + (e.key === "ArrowRight" ? 1 : -1) + spans.length) % spans.length;
    playSpan(next);
    document.getElementById(`moment-${videoId}-${next}`)?.focus();
  }

  return (
    <section aria-label="Player" className="card-lg p-3 sm:p-4">
      <div className="relative overflow-hidden rounded-video bg-video">
        <video
          ref={videoRef}
          key={streamUrl}
          src={streamUrl}
          className="aspect-video w-full rounded-video bg-video"
          controls
          muted
          playsInline
          preload="metadata"
          onTimeUpdate={onTimeUpdate}
          onSeeking={onSeeking}
          aria-label={`Video: ${title}`}
        />
        {span && (
          <span className="pointer-events-none absolute bottom-14 left-3 rounded-[10px] bg-black/70 px-2.5 py-1 text-[13px] font-medium tabular-nums text-white">
            Playing {formatSpan(span)}
            {looping ? " · looping" : ""}
          </span>
        )}
        {modelName && (
          <span className="pointer-events-none absolute right-3 top-3 rounded-[10px] bg-black/60 px-2.5 py-1 text-[12px] font-medium text-white">
            {modelName}
          </span>
        )}
      </div>

      <div className="px-1 pt-4 sm:px-2">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h2 className="text-[21px] font-semibold leading-tight tracking-[-0.015em]">{title}</h2>
          {score !== undefined && score !== null && (
            <span className="text-[15px] font-semibold text-accent-text" aria-label={`${matchPercent(score)} percent match`}>
              {formatMatch(score)}
            </span>
          )}
        </div>
        {meta && <p className="meta mt-1">{meta}</p>}

        <div className="mt-4">
          <Timeline spans={spans} duration={duration} saliency={saliency} activeIndex={active} currentTime={time} onSeek={(t) => { setLooping(false); seekTo(t); }} />
        </div>

        {spans.length > 0 ? (
          <div className="mt-4">
            <h3 className="text-[13px] font-semibold text-text-secondary">Moments</h3>
            <div role="group" aria-label="Predicted moments" className="mt-2 flex flex-wrap gap-2">
              {spans.map((s, i) => (
                <button
                  key={`${s.start}-${s.end}`}
                  id={`moment-${videoId}-${i}`}
                  type="button"
                  tabIndex={i === active ? 0 : -1}
                  aria-pressed={i === active}
                  onClick={() => playSpan(i)}
                  onKeyDown={(e) => onChipKey(e, i)}
                  className={`rounded-pill px-3.5 py-1.5 text-[13px] font-medium tabular-nums transition-colors duration-150 ease-out ${
                    i === active ? "bg-accent-tint text-accent-text" : "bg-bg text-text-nav hover:bg-fill"
                  }`}
                >
                  {formatSpan(s)} · {matchPercent(s.score)}%
                </button>
              ))}
            </div>
          </div>
        ) : (
          <p className="meta mt-4">No moment predicted for this video. Showing it from {formatTime(0)}.</p>
        )}

        {showFeedback && queryId && <FeedbackPrompt queryId={queryId} videoId={videoId} span={span} />}
      </div>
    </section>
  );
}

export function FeedbackPrompt({ queryId, videoId, span }: { queryId: string; videoId: string; span?: Span }) {
  const feedback = useFeedback();
  const [answered, setAnswered] = useState(false);
  useEffect(() => setAnswered(false), [queryId, videoId]);

  if (answered) {
    return (
      <p role="status" className="mt-5 border-t border-hairline-soft pt-4 text-[15px] text-text-secondary">
        Thanks. Your answer helps us evaluate the model.
      </p>
    );
  }
  const send = (relevant: boolean) => {
    feedback.mutate({ query_id: queryId, video_id: videoId, relevant, span: span ?? null });
    setAnswered(true);
  };
  return (
    <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-hairline-soft pt-4">
      <p className="text-[15px] font-medium">Is this the right moment?</p>
      <div className="flex gap-2">
        <button type="button" className="btn-secondary h-9 min-h-0 px-4" onClick={() => send(true)}>
          Yes
        </button>
        <button type="button" className="btn-secondary h-9 min-h-0 px-4" onClick={() => send(false)}>
          No
        </button>
      </div>
    </div>
  );
}