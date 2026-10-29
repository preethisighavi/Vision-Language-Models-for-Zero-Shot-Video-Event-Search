"""Deterministic fake model for UI development and tests.

Embeddings are built from hashed word vectors, so queries that share words with
a video's title land close to it. Everything is seeded from hashes of the
inputs, so the same query always returns the same results.
"""

import hashlib
import math
import random
import re
import time
from typing import Sequence

import numpy as np

from app.models.base import (
    Candidate,
    LocalizeOutput,
    SpanPrediction,
    TextEncoding,
    WindowContext,
    WindowEncoding,
    l2_normalize,
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "at", "to",
    "and", "or", "with", "for", "from", "by", "it", "its", "this", "that", "some",
    "someone", "there", "into", "onto", "up", "down", "out", "over", "his", "her",
    "their", "they", "he", "she", "video", "clip",
}

FEATURE_DIM = 64


def _seed(*parts: str) -> int:
    digest = hashlib.md5("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


class MockAdapter:
    def __init__(
        self,
        model_id: str = "mock",
        embedding_dim: int = 512,
        frames_per_window: int = 12,
        window_seconds: float = 10.0,
        window_stride: float = 5.0,
        supports_rerank: bool = True,
        supports_localize: bool = True,
        noise: float = 0.25,
        salt: str = "",
        latency: bool | None = None,
    ):
        self.id = model_id
        self.embedding_dim = int(embedding_dim)
        self.frames_per_window = int(frames_per_window)
        self.window_seconds = float(window_seconds)
        self.window_stride = float(window_stride)
        self.supports_rerank = bool(supports_rerank)
        self.supports_localize = bool(supports_localize)
        self.noise = float(noise)
        self.salt = salt or model_id
        self._latency = latency

    # ── helpers ──────────────────────────────────────────────────────────
    def _latency_enabled(self) -> bool:
        if self._latency is not None:
            return self._latency
        from app.config import get_settings

        return get_settings().mock_latency

    def _sleep(self, low_ms: float, high_ms: float) -> None:
        if self._latency_enabled():
            time.sleep(random.uniform(low_ms, high_ms) / 1000.0)

    def _vec(self, *parts: str) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.salt, *parts))
        return rng.standard_normal(self.embedding_dim).astype(np.float32)

    def _text_vec(self, text: str) -> np.ndarray:
        words = tokenize(text)
        if not words:
            return l2_normalize(self._vec("text", text))
        total = np.zeros(self.embedding_dim, dtype=np.float32)
        for w in words:
            total += l2_normalize(self._vec("word", w))
        return l2_normalize(total)

    # ── ModelAdapter ─────────────────────────────────────────────────────
    def load(self) -> None:
        return None

    def encode_text(self, texts: Sequence[str]) -> list[TextEncoding]:
        self._sleep(5, 30)
        out = []
        for text in texts:
            words = tokenize(text) or [text.lower()]
            states = np.stack([l2_normalize(self._vec("word", w))[:FEATURE_DIM] for w in words])
            out.append(TextEncoding(embedding=self._text_vec(text), token_states=states.astype(np.float16)))
        return out

    def encode_window(
        self, frames: np.ndarray, times: np.ndarray, context: WindowContext | None = None
    ) -> WindowEncoding:
        video_id = context.video_id if context else "unknown"
        window_id = context.window_id if context else video_id
        title = context.title if context else ""
        base = self._text_vec(title) if title else l2_normalize(self._vec("video", video_id))
        noise = l2_normalize(self._vec("noise", video_id))
        embedding = l2_normalize(base + self.noise * noise)
        t = len(times)
        rng = np.random.default_rng(_seed(self.salt, "frames", window_id))
        features = rng.standard_normal((t, FEATURE_DIM)).astype(np.float16)
        return WindowEncoding(
            embedding=embedding,
            frame_features=features,
            frame_times=np.asarray(times, dtype=np.float32),
        )

    def rerank(self, text: TextEncoding, candidates: Sequence[Candidate]) -> list[float]:
        self._sleep(5, 30)
        key = text.embedding[:8].tobytes().hex()
        scores = []
        for c in candidates:
            jitter = (_seed(key, c.window_id) % 10_000) / 10_000.0 - 0.5
            scores.append(float(c.stage1_score + 0.12 * jitter))
        return scores

    def localize(
        self, text: TextEncoding, frame_features: np.ndarray, frame_times: np.ndarray
    ) -> LocalizeOutput:
        self._sleep(0.5, 3)
        times = np.asarray(frame_times, dtype=np.float32)
        if times.size == 0:
            return LocalizeOutput(spans=[], saliency=None, frame_times=times)
        step = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        lo = max(0.0, float(times.min()) - step / 2)
        hi = float(times.max()) + step / 2
        length = max(hi - lo, 1e-3)

        rng = random.Random(_seed(text.embedding[:16].tobytes().hex(), frame_features[:2].tobytes().hex()))
        c1 = lo + length * rng.uniform(0.2, 0.8)
        s1 = max(length * 0.12, 0.8)
        c2 = lo + length * rng.uniform(0.05, 0.95)
        if abs(c2 - c1) < 2.5 * s1:
            c2 = lo + (c1 - lo + length / 2) % length
        s2 = s1 * 0.6

        def bump(x, c, s, a):
            return a * np.exp(-((x - c) ** 2) / (2 * s * s))

        saliency = np.clip(0.1 + bump(times, c1, s1, 0.85) + bump(times, c2, s2, 0.42), 0, 1)

        half1 = s1 * math.sqrt(2 * math.log(0.85 / 0.5))  # where the main bump crosses 0.6
        top = SpanPrediction(
            start=max(lo, c1 - half1), end=min(hi, c1 + half1), score=round(rng.uniform(0.7, 0.95), 3)
        )
        half2 = max(s2 * 0.9, 0.6)
        weak = SpanPrediction(
            start=max(lo, c2 - half2), end=min(hi, c2 + half2), score=round(rng.uniform(0.3, 0.55), 3)
        )
        return LocalizeOutput(spans=[top, weak], saliency=saliency.astype(np.float32), frame_times=times)

    def fuse(self, stage1: float, rerank: float | None) -> float:
        value = rerank if rerank is not None else stage1
        return float(1.0 / (1.0 + math.exp(-(value - 0.3) * 7.0)))
