import type { Span } from "../api/client";
import { formatSpan, formatTime } from "../lib/format";

const pct = (t: number, duration: number) => `${(Math.min(Math.max(t / Math.max(duration, 0.001), 0), 1) * 100).toFixed(3)}%`;

/** The thin 4 px track used on result cards: only the top span. */
export function MiniTimeline({ span, duration }: { span?: Span; duration: number }) {
  return (
    <div className="relative h-1 w-full overflow-hidden rounded-pill bg-fill" aria-hidden>
      {span && (
        <div
          className="absolute inset-y-0 rounded-pill bg-accent"
          style={{ left: pct(span.start, duration), width: `calc(${pct(span.end, duration)} - ${pct(span.start, duration)})` }}
        />
      )}
    </div>
  );
}

/** The 8 px player timeline: every span, an optional saliency strip and the current position. */
export function Timeline({
  spans,
  duration,
  saliency,
  activeIndex = 0,
  currentTime,
  onSeek,
}: {
  spans: Span[];
  duration: number;
  saliency?: number[] | null;
  activeIndex?: number;
  currentTime?: number;
  onSeek?: (t: number) => void;
}) {
  return (
    <div>
      <div
        className={`relative h-2 w-full rounded-pill bg-fill ${onSeek ? "cursor-pointer" : ""}`}
        onClick={(e) => {
          if (!onSeek) return;
          const rect = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - rect.left) / rect.width) * duration);
        }}
        role="img"
        aria-label={
          spans.length
            ? `Timeline with ${spans.length} predicted moment${spans.length === 1 ? "" : "s"}: ${spans.map(formatSpan).join(", ")}`
            : "Timeline"
        }
      >
        {spans.map((s, i) => (
          <div
            key={`${s.start}-${s.end}`}
            className="absolute inset-y-0 rounded-pill transition-colors duration-200"
            style={{
              left: pct(s.start, duration),
              width: `calc(${pct(s.end, duration)} - ${pct(s.start, duration)})`,
              background: i === activeIndex ? "var(--accent)" : "var(--accent-secondary-span)",
            }}
          />
        ))}
        {currentTime !== undefined && (
          <div
            className="absolute -top-[3px] h-[14px] w-[2px] -translate-x-1/2 rounded-pill bg-text"
            style={{ left: pct(currentTime, duration) }}
            aria-hidden
          />
        )}
      </div>
      {saliency && saliency.length > 0 && (
        <div className="mt-1.5 flex h-2 w-full overflow-hidden rounded-pill" aria-hidden>
          {saliency.map((v, i) => (
            <div key={i} className="h-full flex-1" style={{ background: "var(--accent)", opacity: Math.max(0.06, v) }} />
          ))}
        </div>
      )}
      <div className="mt-1.5 flex justify-between text-[12px] tabular-nums text-text-secondary">
        <span>0:00</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
