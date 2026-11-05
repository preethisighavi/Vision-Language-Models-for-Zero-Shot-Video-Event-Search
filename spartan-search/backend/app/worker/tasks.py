"""RQ tasks. Each task re-initialises the DB so it can run in a separate process."""

from app.db.session import init_db
from app.services import generation_service, ingest_service


def ingest_video(video_id: str, source_path: str, job_id: str) -> None:
    init_db()
    ingest_service.ingest_video(video_id, source_path, job_id)


def reindex(model_id: str, collection_id: str, job_id: str) -> None:
    init_db()
    ingest_service.reindex_collection(model_id, collection_id, job_id)


def generate_video(video_id: str, prompt: str, model: str, seed: int, job_id: str) -> None:
    init_db()
    generation_service.generate_video(video_id, prompt, model, seed, job_id)
