"""RQ tasks. Each task re-initialises the DB so it can run in a separate process."""

from app.db.session import init_db
from app.services import ingest_service


def ingest_video(video_id: str, source_path: str, job_id: str) -> None:
    init_db()
    ingest_service.ingest_video(video_id, source_path, job_id)


def reindex(model_id: str, collection_id: str, job_id: str) -> None:
    init_db()
    ingest_service.reindex_collection(model_id, collection_id, job_id)
