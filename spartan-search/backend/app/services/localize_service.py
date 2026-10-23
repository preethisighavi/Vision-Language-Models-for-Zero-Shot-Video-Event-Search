"""Turns adapter localization output into clean spans and a per-second saliency strip."""

import math

import numpy as np
from sqlmodel import Session, select

from app.db.tables import Window
from app.models.base import ModelAdapter, SpanPrediction, TextEncoding
from app.store.feature_store import FeatureStore

MAX_SPANS = 3
MIN_SPAN_SECONDS = 1.0
NMS_IOU = 0.5
SALIENCY_FPS = 1.0


def _iou(a: SpanPrediction, b: SpanPrediction) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def clean_spans(spans: list[SpanPrediction], duration: float) -> list[SpanPrediction]:
    fixed = []
    for s in spans:
        start, end = max(0.0, min(s.start, s.end)), min(duration, max(s.start, s.end))
        if end - start < MIN_SPAN_SECONDS:
            centre = (start + end) / 2
            start = max(0.0, centre - MIN_SPAN_SECONDS / 2)
            end = min(duration, start + MIN_SPAN_SECONDS)
            start = max(0.0, end - MIN_SPAN_SECONDS)
        if end > start:
            fixed.append(SpanPrediction(start, end, float(s.score)))
    fixed.sort(key=lambda s: s.score, reverse=True)

    kept: list[SpanPrediction] = []
    for s in fixed:  # non-maximum suppression
        if all(_iou(s, k) <= NMS_IOU for k in kept):
            kept.append(s)

    merged: list[SpanPrediction] = []
    for s in kept:  # merge remaining overlaps into the higher-scoring span
        for i, m in enumerate(merged):
            if s.start < m.end and m.start < s.end:
                merged[i] = SpanPrediction(min(m.start, s.start), max(m.end, s.end), max(m.score, s.score))
                break
        else:
            merged.append(s)
    merged.sort(key=lambda s: s.score, reverse=True)
    return [SpanPrediction(round(s.start, 2), round(s.end, 2), round(s.score, 3)) for s in merged[:MAX_SPANS]]


def saliency_strip(saliency: np.ndarray | None, times: np.ndarray, duration: float) -> list[float] | None:
    if saliency is None or len(saliency) == 0 or duration <= 0:
        return None
    n = max(1, math.ceil(duration * SALIENCY_FPS))
    grid = (np.arange(n) + 0.5) / SALIENCY_FPS
    order = np.argsort(times)
    values = np.interp(grid, times[order], saliency[order], left=0.0, right=0.0)
    covered = (grid >= times.min() - 1.0) & (grid <= times.max() + 1.0)
    values = np.where(covered, values, 0.0)
    return [round(float(v), 3) for v in np.clip(values, 0, 1)]


def gather_features(model_id: str, windows: list[Window]) -> tuple[np.ndarray, np.ndarray] | None:
    store = FeatureStore()
    found = store.get_many(model_id, [w.id for w in windows])
    if not found:
        return None
    feats = np.concatenate([found[w.id][0] for w in windows if w.id in found])
    times = np.concatenate([found[w.id][1] for w in windows if w.id in found]).astype(np.float32)
    order = np.argsort(times, kind="stable")
    times, feats = times[order], feats[order]
    keep = np.concatenate([[True], np.diff(times) > 1e-3]) if len(times) > 1 else np.ones(len(times), bool)
    return feats[keep], times[keep]


def neighbour_windows(session: Session, video_id: str, best_window_id: str | None) -> list[Window]:
    windows = session.exec(select(Window).where(Window.video_id == video_id).order_by(Window.start)).all()
    if not best_window_id or len(windows) <= 1:
        return list(windows)
    idx = next((i for i, w in enumerate(windows) if w.id == best_window_id), 0)
    return list(windows[max(0, idx - 1): idx + 2])


def grounded_answer(query: str, spans: list) -> str:
    """A retrieval answer: the question, plus the moment in the video that supports it."""
    if not spans:
        return f"I couldn't find “{query}” in this video."

    def clock(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60}:{total % 60:02d}"

    top = spans[0]
    text = (
        f"The closest moment for “{query}” is {clock(top.start)}–{clock(top.end)} "
        f"({round(top.score * 100)}% match)."
    )
    if len(spans) > 1:
        nxt = spans[1]
        text += f" There is also a weaker match at {clock(nxt.start)}–{clock(nxt.end)}."
    return text


def localize(adapter: ModelAdapter, model_id: str, text: TextEncoding, windows: list[Window], duration: float):
    gathered = gather_features(model_id, windows)
    if gathered is None:
        return [], None
    feats, times = gathered
    out = adapter.localize(text, feats, times)
    spans = clean_spans(list(out.spans), duration)
    sal_times = out.frame_times if out.frame_times is not None and len(out.frame_times) else times
    return spans, saliency_strip(out.saliency, np.asarray(sal_times, np.float32), duration)
