from typing import Any, Protocol, Sequence

import numpy as np

Hit = tuple[str, float, dict[str, Any]]


class VectorIndex(Protocol):
    """One index per (model_id, collection_id). Never mix models in one index."""

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None: ...

    def search(self, vector: np.ndarray, k: int, filter: dict[str, Any] | None = None) -> list[Hit]: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...
