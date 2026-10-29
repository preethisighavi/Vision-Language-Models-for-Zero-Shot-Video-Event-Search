"""Read-side helpers shared by several routers: models, collections, index sizes."""

from sqlmodel import Session, func, select

from app.db.tables import Collection, Video
from app.errors import ApiError
from app.index.factory import get_index
from app.models.base import ModelAdapter
from app.models.registry import get_registry
from app.schemas.api import CollectionInfo, ModelInfo

DEFAULT_COLLECTIONS = [
    ("msrvtt-1ka", "MSR-VTT 1k-A", "Clips from the MSR-VTT 1k-A test set."),
    ("msvd-test", "MSVD test", "Clips from the MSVD test split."),
    ("uploads", "Your uploads", "Videos uploaded through this site."),
]


def ensure_default_collections(session: Session) -> None:
    for cid, name, description in DEFAULT_COLLECTIONS:
        if session.get(Collection, cid) is None:
            session.add(Collection(id=cid, name=name, description=description))
    session.commit()


def model_infos() -> list[ModelInfo]:
    registry = get_registry()
    infos = []
    for entry in registry.entries():
        adapter = registry.peek(entry.id)
        infos.append(ModelInfo(
            id=entry.id,
            name=entry.name,
            description=entry.description,
            embedding_dim=int(getattr(adapter, "embedding_dim", 0) or 0),
            supports_rerank=bool(getattr(adapter, "supports_rerank", False)),
            supports_localize=bool(getattr(adapter, "supports_localize", False)),
            default=entry.id == registry.default_id,
            status="error" if entry.status == "error" else ("loading" if entry.status == "loading" else "ready"),
        ))
    return infos


def index_sizes(session: Session) -> dict[str, dict[str, int]]:
    collections = [c.id for c in session.exec(select(Collection)).all()]
    return {m: {c: get_index(m, c).count() for c in collections} for m in get_registry().enabled_ids()}


def collection_infos(session: Session) -> list[CollectionInfo]:
    counts = dict(session.exec(
        select(Video.collection, func.count(Video.id)).where(Video.status == "ready").group_by(Video.collection)
    ).all())
    models = get_registry().enabled_ids()
    infos = []
    for c in session.exec(select(Collection)).all():
        infos.append(CollectionInfo(
            id=c.id,
            name=c.name,
            description=c.description,
            video_count=int(counts.get(c.id, 0)),
            indexed_models=[m for m in models if get_index(m, c.id).count() > 0],
        ))
    return infos


def require_model(model_id: str) -> ModelAdapter:
    registry = get_registry()
    if not registry.has(model_id):
        raise ApiError(404, "unknown_model", f"Model '{model_id}' does not exist or is disabled.")
    try:
        return registry.get(model_id)
    except Exception:
        raise ApiError(409, "model_not_ready", f"Model '{model_id}' failed to load.")


def require_collection(session: Session, collection_id: str) -> Collection:
    collection = session.get(Collection, collection_id)
    if collection is None:
        raise ApiError(404, "unknown_collection", f"Collection '{collection_id}' does not exist.")
    return collection


def require_index(model_id: str, collection_id: str) -> None:
    if get_index(model_id, collection_id).count() == 0:
        raise ApiError(
            409, "index_not_ready",
            f"The {collection_id} library is not indexed for {model_id} yet. Run a reindex or try another model.",
        )
