"""Spartan VLM adapter skeleton (two stages: dense retrieval, then fusion re-rank + moments).

To enable: implement the methods below, set `enabled: true` for spartan-v1 in
models.yaml, and run POST /api/v1/admin/reindex for each collection.
"""

from typing import Sequence

import numpy as np

from app.models.base import Candidate, LocalizeOutput, TextEncoding, WindowContext, WindowEncoding


class SpartanAdapter:
    embedding_dim = 512
    window_seconds = 10.0
    window_stride = 5.0
    supports_rerank = True
    supports_localize = True

    def __init__(self, model_id: str = "spartan-v1", checkpoint: str = "", frames_per_window: int = 12,
                 rerank_k: int = 20, device: str = "cpu"):
        self.id = model_id
        self.checkpoint = checkpoint
        self.frames_per_window = int(frames_per_window)
        self.rerank_k = int(rerank_k)  # the search service re-ranks this many stage-1 candidates
        self.device = device

    def load(self) -> None:
        # TODO(team): build the Spartan model (video trunk, text encoder, fusion head, span decoder) and load
        # self.checkpoint. Put it in eval mode on self.device.
        raise NotImplementedError("TODO(team): load Spartan weights")

    def encode_text(self, texts: Sequence[str]) -> list[TextEncoding]:
        # TODO(team): run the text encoder. `embedding` = pooled 512-d, L2-normalised (stage 1).
        # `token_states` = per-token hidden states (L, d) as float16; rerank() and localize() read them.
        raise NotImplementedError("TODO(team): Spartan text encoding")

    def encode_window(self, frames: np.ndarray, times: np.ndarray, context: WindowContext | None = None) -> WindowEncoding:
        # TODO(team): frames are (T, 224, 224, 3) uint8 RGB from segment sampling. Run the space-time trunk.
        # `embedding` = the query-independent global video embedding (512-d, L2-normalised) for the index.
        # `frame_features` = per-frame (or merged-token) features (T, d) float16 for the fusion head.
        raise NotImplementedError("TODO(team): Spartan video encoding")

    def rerank(self, text: TextEncoding, candidates: Sequence[Candidate]) -> list[float]:
        # TODO(team): for each candidate, run the cross-modal fusion head over candidate.frame_features and
        # text.token_states and return its matching logit. Batch the candidates for speed.
        raise NotImplementedError("TODO(team): Spartan fusion re-rank")

    def localize(self, text: TextEncoding, frame_features: np.ndarray, frame_times: np.ndarray) -> LocalizeOutput:
        # TODO(team): run the span decoder + saliency head over the fused per-frame features. Convert the
        # normalised (centre, width) predictions to seconds using frame_times. Return spans sorted by score
        # and per-frame saliency (same length as frame_times).
        raise NotImplementedError("TODO(team): Spartan span decoder")

    def fuse(self, stage1: float, rerank: float | None) -> float:
        # TODO(team): calibrate on the val split. A sigmoid of the re-rank logit is a reasonable default.
        value = rerank if rerank is not None else (stage1 - 0.2) * 10.0
        return float(1.0 / (1.0 + np.exp(-value)))
