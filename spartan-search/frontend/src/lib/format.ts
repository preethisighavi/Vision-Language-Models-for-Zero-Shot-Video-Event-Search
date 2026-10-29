/** 65.4 → "1:05". Negative or missing values render as 0:00. */
export function formatTime(seconds: number | null | undefined): string {
  const total = Math.max(0, Math.floor(seconds ?? 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** A span as "0:05 – 0:11" (en dash with spaces). */
export function formatSpan(span: { start: number; end: number }): string {
  return `${formatTime(span.start)} – ${formatTime(span.end)}`;
}

/** 0.913 → 91 */
export function matchPercent(score: number | null | undefined): number {
  return Math.round(Math.min(Math.max(score ?? 0, 0), 1) * 100);
}

export function formatMatch(score: number | null | undefined): string {
  return `${matchPercent(score)}% match`;
}

export function formatDuration(seconds: number): string {
  return formatTime(seconds);
}

/** Recall-style fractions as percentages with one decimal. */
export function formatMetric(metric: string, value: number): string {
  if (metric.startsWith("R@") || metric.startsWith("R1@") || metric === "mIoU") return `${(value * 100).toFixed(1)}%`;
  if (metric === "MRR") return value.toFixed(3);
  if (metric === "latency_ms") return `${value < 10 ? value.toFixed(1) : Math.round(value)} ms`;
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function plural(n: number, word: string): string {
  return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
}
