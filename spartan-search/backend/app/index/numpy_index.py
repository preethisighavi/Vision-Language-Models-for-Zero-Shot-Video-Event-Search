"""Exact cosine search over L2-normalised vectors, persisted to .npy + .json."""

import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.index.base import Hit


class NumpyIndex:
    def __init__(self, directory: Path, dim: int | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._vec_path = self.directory / "vectors.npy"
        self._meta_path = self.directory / "meta.json"
        self._lock = threading.RLock()
        self._mtime: tuple[int, int] | None = None
        self.dim = dim
        self._ids: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._vectors = np.zeros((0, dim or 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self._meta_path.exists():
            return
        stat = self._meta_path.stat()
        mtime = (stat.st_mtime_ns, stat.st_size)
        if mtime == self._mtime:
            return
        with open(self._meta_path, encoding="utf-8") as f:
            payload = json.load(f)
        vectors = np.load(self._vec_path) if self._vec_path.exists() else np.zeros((0, 0), np.float32)
        self._ids = payload["ids"]
        self._meta = payload["meta"]
        self._vectors = vectors.astype(np.float32)
        self.dim = self._vectors.shape[1] if self._vectors.size else self.dim
        self._mtime = mtime

    def _save(self) -> None:
        tmp_vec = self.directory / "vectors.tmp.npy"
        tmp_meta = self.directory / "meta.tmp.json"
        np.save(tmp_vec, self._vectors)
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "meta": self._meta}, f)
        os.replace(tmp_vec, self._vec_path)
        os.replace(tmp_meta, self._meta_path)
        stat = self._meta_path.stat()
        self._mtime = (stat.st_mtime_ns, stat.st_size)

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        with self._lock:
            self._load()
            position = {id_: i for i, id_ in enumerate(self._ids)}
            if self._vectors.size == 0:
                self._vectors = np.zeros((0, vectors.shape[1]), dtype=np.float32)
            new_rows, new_ids, new_meta = [], [], []
            for id_, vec, meta in zip(ids, vectors, metadata):
                if id_ in position:
                    self._vectors[position[id_]] = vec
                    self._meta[position[id_]] = dict(meta)
                else:
                    new_rows.append(vec)
                    new_ids.append(id_)
                    new_meta.append(dict(meta))
            if new_rows:
                self._vectors = np.vstack([self._vectors, np.stack(new_rows)])
                self._ids.extend(new_ids)
                self._meta.extend(new_meta)
            self._save()

    def search(self, vector: np.ndarray, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        with self._lock:
            self._load()
            if not self._ids:
                return []
            scores = self._vectors @ np.asarray(vector, dtype=np.float32)
            if filter:
                mask = np.array(
                    [all(m.get(key) == val for key, val in filter.items()) for m in self._meta], dtype=bool
                )
                scores = np.where(mask, scores, -np.inf)
            k = min(k, len(self._ids))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._ids[i], float(scores[i]), self._meta[i]) for i in top if np.isfinite(scores[i])]

    def delete(self, ids: Sequence[str]) -> None:
        drop = set(ids)
        with self._lock:
            self._load()
            keep = [i for i, id_ in enumerate(self._ids) if id_ not in drop]
            if len(keep) == len(self._ids):
                return
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self._vectors.shape[1]), np.float32)
            self._ids = [self._ids[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._save()

    def count(self) -> int:
        with self._lock:
            self._load()
            return len(self._ids)
