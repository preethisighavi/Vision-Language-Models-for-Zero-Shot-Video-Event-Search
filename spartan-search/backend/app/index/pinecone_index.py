"""Pinecone implementation of VectorIndex (stub). Select with INDEX_BACKEND=pinecone.

Our existing index (pipeline/pinecone_manager.py) is serverless, AWS us-east-1, cosine, 512-d.
One namespace per (model, collection) keeps models separate inside one Pinecone index.
"""

from typing import Any, Sequence

import numpy as np

from app.index.base import Hit


class PineconeIndex:
    def __init__(self, index_name: str, namespace: str):
        # TODO(team): `pip install pinecone`, then:
        #   pc = Pinecone(api_key=get_settings().pinecone_api_key)
        #   self._index = pc.Index(index_name); self._ns = namespace
        raise NotImplementedError("TODO(team): PineconeIndex")

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        # self._index.upsert(vectors=[(i, v.tolist(), m) for i, v, m in zip(ids, vectors, metadata)], namespace=self._ns)
        raise NotImplementedError

    def search(self, vector: np.ndarray, k: int, filter: dict[str, Any] | None = None) -> list[Hit]:
        # r = self._index.query(vector=vector.tolist(), top_k=k, filter=filter, include_metadata=True, namespace=self._ns)
        # return [(m.id, m.score, m.metadata) for m in r.matches]
        raise NotImplementedError

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        # self._index.describe_index_stats()["namespaces"].get(self._ns, {}).get("vector_count", 0)
        raise NotImplementedError
