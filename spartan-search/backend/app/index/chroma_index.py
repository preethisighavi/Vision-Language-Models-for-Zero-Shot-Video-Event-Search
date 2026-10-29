"""ChromaDB implementation of VectorIndex (stub). Select with INDEX_BACKEND=chroma."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.index.base import Hit


class ChromaIndex:
    def __init__(self, directory: Path, collection_name: str):
        # TODO(team): `pip install chromadb`, then:
        #   client = chromadb.PersistentClient(path=str(directory))
        #   self._col = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        raise NotImplementedError("TODO(team): ChromaIndex")

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        # self._col.upsert(ids=list(ids), embeddings=vectors.tolist(), metadatas=list(metadata))
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        # r = self._col.query(query_embeddings=[vector.tolist()], n_results=k, where=filter or None)
        # Chroma returns cosine *distance*; convert with score = 1 - distance so higher is better.
        raise NotImplementedError

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError
