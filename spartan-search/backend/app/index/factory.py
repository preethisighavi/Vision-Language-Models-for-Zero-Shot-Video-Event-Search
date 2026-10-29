import threading

from app.config import get_settings
from app.index.base import VectorIndex

_cache: dict[tuple[str, str], VectorIndex] = {}
_lock = threading.Lock()


def get_index(model_id: str, collection_id: str) -> VectorIndex:
    key = (model_id, collection_id)
    with _lock:
        if key not in _cache:
            _cache[key] = _create(model_id, collection_id)
        return _cache[key]


def _create(model_id: str, collection_id: str) -> VectorIndex:
    settings = get_settings()
    backend = settings.index_backend.lower()
    if backend == "numpy":
        from app.index.numpy_index import NumpyIndex

        return NumpyIndex(settings.index_dir / model_id / collection_id)
    if backend == "chroma":
        from app.index.chroma_index import ChromaIndex

        return ChromaIndex(settings.index_dir / "chroma", f"{model_id}__{collection_id}")
    if backend == "pinecone":
        from app.index.pinecone_index import PineconeIndex

        return PineconeIndex(settings.pinecone_index, namespace=f"{model_id}__{collection_id}")
    raise ValueError(f"unknown INDEX_BACKEND: {settings.index_backend}")


def reset_index_cache() -> None:
    with _lock:
        _cache.clear()
