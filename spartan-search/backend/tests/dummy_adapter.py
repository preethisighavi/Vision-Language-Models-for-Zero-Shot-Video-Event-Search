"""A minimal third-party-style adapter, used to prove the plug-in boundary.

It lives outside app/ on purpose and only implements the ModelAdapter protocol.
"""

from typing import Sequence

import numpy as np

from app.models.base import LocalizeOutput, SpanPrediction, TextEncoding, WindowEncoding


class DummyAdapter:
    embedding_dim = 8
    frames_per_window = 4
    window_seconds = 10.0
    window_stride = 5.0
    supports_rerank = False
    supports_localize = True

    def __init__(self, model_id: str = "dummy"):
        self.id = model_id
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def encode_text(self, texts: Sequence[str]) -> list[TextEncoding]:
        return [TextEncoding(embedding=np.eye(8, dtype=np.float32)[len(t) % 8]) for t in texts]

    def encode_window(self, frames, times, context=None) -> WindowEncoding:
        # A real model reads pixels; this one uses the mean brightness of the frames.
        bucket = int(frames.mean()) % 8
        return WindowEncoding(
            embedding=np.eye(8, dtype=np.float32)[bucket],
            frame_features=np.ones((len(times), 2), dtype=np.float16),
            frame_times=np.asarray(times, dtype=np.float32),
        )

    def rerank(self, text, candidates):
        raise NotImplementedError

    def localize(self, text, frame_features, frame_times) -> LocalizeOutput:
        return LocalizeOutput(
            spans=[SpanPrediction(float(frame_times[0]), float(frame_times[-1]), 0.9)],
            saliency=np.linspace(0, 1, len(frame_times), dtype=np.float32),
            frame_times=frame_times,
        )

    def fuse(self, stage1: float, rerank: float | None) -> float:
        return float(np.clip((stage1 + 1) / 2, 0, 1))
