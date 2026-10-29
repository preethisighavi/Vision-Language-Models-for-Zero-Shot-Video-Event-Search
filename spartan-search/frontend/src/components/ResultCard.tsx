import { mediaUrl, type SearchResult } from "../api/client";
import { formatMatch, formatSpan, formatTime } from "../lib/format";
import { Thumb } from "./Thumb";
import { MiniTimeline } from "./Timeline";

export function ResultCard({
  result,
  selected,
  collectionName,
  onSelect,
}: {
  result: SearchResult;
  selected: boolean;
  collectionName?: string;
  onSelect: () => void;
}) {
  const top = result.spans[0];
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className={`flex w-full gap-4 rounded-card border bg-surface p-[14px] text-left shadow-sm transition-all duration-200 ease-out hover:shadow-md ${
          selected ? "border-accent" : "border-transparent"
        }`}
        style={selected ? { boxShadow: "0 0 0 3px var(--accent-ring)" } : undefined}
      >
        <Thumb src={mediaUrl(result.thumbnail_url)} className="w-[132px] shrink-0 sm:w-[184px]">
          {top && (
            <span className="absolute bottom-1.5 left-1.5 rounded-[8px] bg-black/70 px-1.5 py-0.5 text-[12px] font-medium tabular-nums text-white">
              {formatSpan(top)}
            </span>
          )}
        </Thumb>
        <span className="flex min-w-0 flex-1 flex-col py-0.5">
          <span className="text-[13px] font-semibold text-accent-text">
            {result.rank}. {formatMatch(result.score)}
          </span>
          <span className="mt-0.5 line-clamp-2 text-[19px] font-semibold leading-snug tracking-[-0.015em]">{result.title}</span>
          <span className="meta mt-1">
            {collectionName ?? result.collection} · {formatTime(result.duration)}
          </span>
          <span className="mt-auto pt-3">
            <MiniTimeline span={top} duration={result.duration} />
          </span>
        </span>
      </button>
    </li>
  );
}
