"""Per-window frame features on disk, keyed by (model_id, window_id)."""

import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import get_settings


class FeatureStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().features_dir)

    def _path(self, model_id: str, window_id: str) -> Path:
        return self.root / model_id / f"{window_id}.npz"

    def put(self, model_id: str, window_id: str, features: np.ndarray, times: np.ndarray) -> None:
        path = self._path(model_id, window_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, features=np.asarray(features), times=np.asarray(times, dtype=np.float32))
        tmp.replace(path)

    def get(self, model_id: str, window_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        path = self._path(model_id, window_id)
        if not path.exists():
            return None
        with np.load(path) as data:
            return data["features"], data["times"]

    def get_many(self, model_id: str, window_ids: Sequence[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for wid in window_ids:
            item = self.get(model_id, wid)
            if item is not None:
                out[wid] = item
        return out

    def delete(self, model_id: str, window_ids: Sequence[str]) -> None:
        for wid in window_ids:
            self._path(model_id, wid).unlink(missing_ok=True)

    def drop_model(self, model_id: str) -> None:
        shutil.rmtree(self.root / model_id, ignore_errors=True)
