"""The plug-in boundary between the platform and the models.

The platform owns decoding, sampling, windowing, the index, the feature store,
grouping, logging and the API. An adapter owns only the maths.
"""

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class TextEncoding:
    embedding: np.ndarray  # (D,) float32, L2-normalised, for stage-1 search
    token_states: np.ndarray | None = None  # (L, d) float16, for re-rank / localize


@dataclass
class WindowEncoding:
    embedding: np.ndarray  # (D,) float32, L2-normalised, stored in the vector index
    frame_features: np.ndarray  # (T, d) float16, stored in the feature store
    frame_times: np.ndarray  # (T,) float32 seconds, absolute within the video


@dataclass
class WindowContext:
    """Metadata about the window being encoded. Real models should ignore it;
    the mock uses it to produce sensible orderings without looking at pixels."""

    video_id: str
    window_id: str
    title: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class Candidate:
    window_id: str
    video_id: str
    start: float
    end: float
    stage1_score: float
    frame_features: np.ndarray
    frame_times: np.ndarray


@dataclass
class SpanPrediction:
    start: float
    end: float
    score: float


@dataclass
class LocalizeOutput:
    spans: list[SpanPrediction]  # sorted by score desc
    saliency: np.ndarray | None  # per frame, same length as frame_times
    frame_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@runtime_checkable
class ModelAdapter(Protocol):
    id: str
    embedding_dim: int
    frames_per_window: int  # T sampled per window for encode_window
    window_seconds: float  # e.g. 10.0 for uploads; trimmed clips = whole clip
    window_stride: float  # e.g. 5.0
    supports_rerank: bool
    supports_localize: bool

    def load(self) -> None: ...

    def encode_text(self, texts: Sequence[str]) -> list[TextEncoding]: ...

    def encode_window(
        self, frames: np.ndarray, times: np.ndarray, context: WindowContext | None = None
    ) -> WindowEncoding:
        """frames: (T, H, W, 3) uint8 RGB, already sampled by the platform."""
        ...

    def rerank(self, text: TextEncoding, candidates: Sequence[Candidate]) -> list[float]:
        """One score per candidate (higher is better). Only called if supports_rerank."""
        ...

    def localize(
        self, text: TextEncoding, frame_features: np.ndarray, frame_times: np.ndarray
    ) -> LocalizeOutput:
        """Only called if supports_localize."""
        ...

    def fuse(self, stage1: float, rerank: float | None) -> float:
        """Model-specific score fusion. Must return 0-1 for display."""
        ...


def l2_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return (v / np.maximum(norm, 1e-12)).astype(np.float32)
