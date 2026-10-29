"""CLIP ViT-B/32 adapter.

`load()` uses the pretrained `openai/clip-vit-base-patch32` weights. If `checkpoint`
points at a fine-tuned file, those weights replace the pretrained ones.
"""

from pathlib import Path
from typing import Sequence

import numpy as np

from app.models.base import Candidate, LocalizeOutput, SpanPrediction, TextEncoding, WindowContext, WindowEncoding


class ClipAdapter:
    embedding_dim = 512
    window_seconds = 10.0
    window_stride = 5.0
    supports_rerank = False
    supports_localize = True

    def __init__(self, model_id: str = "clip-b32", checkpoint: str = "", frames_per_window: int = 8,
                 model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.id = model_id
        self.checkpoint = checkpoint
        self.frames_per_window = int(frames_per_window)
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None
        self._processor = None
        self.using_checkpoint = False

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        model = CLIPModel.from_pretrained(self.model_name)
        path = Path(self.checkpoint) if self.checkpoint else None
        if path is not None and path.is_file():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
                state = blob["model"]
            elif isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
                state = blob["state_dict"]
            else:
                state = blob
            state = {k.removeprefix("module."): v for k, v in state.items() if hasattr(v, "shape")}
            model.load_state_dict(state, strict=False)
            self.using_checkpoint = True
        self._model = model.to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def encode_text(self, texts: Sequence[str]) -> list[TextEncoding]:
        self.load()
        torch = self._torch
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self._model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            pooled = self._model.text_projection(outputs.pooler_output)
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
        vectors = pooled.detach().cpu().float().numpy()
        return [TextEncoding(embedding=v.astype(np.float32)) for v in vectors]

    def encode_window(self, frames: np.ndarray, times: np.ndarray, context: WindowContext | None = None) -> WindowEncoding:
        self.load()
        torch = self._torch
        images = [frame for frame in frames] if len(frames) else [np.zeros((224, 224, 3), np.uint8)]
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
            projected = self._model.visual_projection(outputs.pooler_output)
            projected = projected / projected.norm(p=2, dim=-1, keepdim=True)
        per_frame = projected.detach().cpu().float().numpy().astype(np.float32)
        pooled = per_frame.mean(axis=0)
        pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-12)
        return WindowEncoding(
            embedding=pooled.astype(np.float32),
            frame_features=per_frame.astype(np.float16),
            frame_times=np.asarray(times, dtype=np.float32),
        )

    def rerank(self, text: TextEncoding, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError("CLIP has no re-rank stage (supports_rerank=False)")

    def localize(self, text: TextEncoding, frame_features: np.ndarray, frame_times: np.ndarray) -> LocalizeOutput:
        times = np.asarray(frame_times, dtype=np.float32)
        features = np.asarray(frame_features, dtype=np.float32)
        if times.size == 0 or features.size == 0:
            return LocalizeOutput(spans=[], saliency=None, frame_times=times)
        features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
        sims = features @ np.asarray(text.embedding, dtype=np.float32)
        smooth = np.convolve(sims, np.array([0.25, 0.5, 0.25]), mode="same")
        span = float(smooth.max() - smooth.min())
        saliency = (smooth - smooth.min()) / span if span > 1e-6 else np.ones_like(smooth)
        return LocalizeOutput(
            spans=_spans_from_saliency(saliency, times),
            saliency=saliency.astype(np.float32),
            frame_times=times,
        )

    def fuse(self, stage1: float, rerank: float | None) -> float:
        # CLIP cosine is usually about 0.15 to 0.40. Map that range onto 0-1 for the match percent.
        return float(np.clip((stage1 - 0.15) / 0.25, 0.0, 1.0))


def _spans_from_saliency(saliency: np.ndarray, times: np.ndarray, threshold: float = 0.6) -> list[SpanPrediction]:
    above = saliency >= threshold
    spans: list[SpanPrediction] = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append(_span(saliency, times, start, i - 1))
            start = None
    if start is not None:
        spans.append(_span(saliency, times, start, len(saliency) - 1))
    spans.sort(key=lambda s: s.score, reverse=True)
    return spans


def _span(saliency: np.ndarray, times: np.ndarray, first: int, last: int) -> SpanPrediction:
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return SpanPrediction(
        start=max(0.0, float(times[first]) - step / 2),
        end=float(times[last]) + step / 2,
        score=float(saliency[first:last + 1].mean()),
    )
