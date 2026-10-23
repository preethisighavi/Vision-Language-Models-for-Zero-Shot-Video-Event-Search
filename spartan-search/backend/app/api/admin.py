import json
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Response
from sqlmodel import Session, select

from app.config import get_settings
from app.db.session import get_session
from app.db.tables import Collection, Job, Query, Video, Window
from app.errors import ApiError
from app.models.registry import get_registry
from app.schemas.api import CollectionCreate, CollectionInfo, ErrorResponse, ImportRequest, JobAccepted, ReindexRequest, Stats
from app.security import require_admin
from app.services import catalog
from app.services.ingest_service import remove_vectors
from app.services.library_service import import_folder
from app.store.media_store import MediaStore
from app.worker import tasks
from app.worker.queue import enqueue

router = APIRouter(tags=["admin"], responses={401: {"model": ErrorResponse}})


@router.delete("/videos/{video_id}", status_code=204, dependencies=[Depends(require_admin)],
               responses={404: {"model": ErrorResponse}})
def delete_video(video_id: str, session: Session = Depends(get_session)) -> Response:
    video = session.get(Video, video_id)
    if video is None:
        raise ApiError(404, "not_found", f"Video '{video_id}' does not exist.")
    remove_vectors(video_id)
    for w in session.exec(select(Window).where(Window.video_id == video_id)).all():
        session.delete(w)
    session.commit()  # windows reference the video, so they must go first
    session.delete(video)
    session.commit()
    MediaStore().delete_video(video_id)
    return Response(status_code=204)


@router.post("/admin/reindex", status_code=202, response_model=JobAccepted, dependencies=[Depends(require_admin)],
             responses={404: {"model": ErrorResponse}})
def reindex(req: ReindexRequest, background: BackgroundTasks, session: Session = Depends(get_session)) -> JobAccepted:
    if not get_registry().has(req.model):
        raise ApiError(404, "unknown_model", f"Model '{req.model}' does not exist or is disabled.")
    catalog.require_collection(session, req.collection)
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    session.add(Job(id=job_id, kind="reindex", status="queued", step_detail=f"Reindex {req.collection} with {req.model}"))
    session.commit()
    enqueue(background, tasks.reindex, req.model, req.collection, job_id)
    return JobAccepted(job_id=job_id)


@router.get("/admin/stats", response_model=Stats, dependencies=[Depends(require_admin)])
def stats(session: Session = Depends(get_session)) -> Stats:
    rows = session.exec(select(Query.model, Query.latency_json)).all()
    totals: dict[str, list[float]] = {}
    for model, latency in rows:
        totals.setdefault(model, []).append(float(json.loads(latency or "{}").get("total", 0)))
    every = [t for ts in totals.values() for t in ts]

    def pct(values: list[float], q: float) -> float:
        return round(float(np.percentile(values, q)), 1) if values else 0.0

    return Stats(
        queries_total=len(every), p50_ms=pct(every, 50), p95_ms=pct(every, 95),
        by_model={m: {"queries": len(ts), "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95)} for m, ts in totals.items()},
    )


@router.post("/admin/collections", status_code=201, response_model=CollectionInfo, dependencies=[Depends(require_admin)],
             responses={409: {"model": ErrorResponse}})
def create_collection(req: CollectionCreate, session: Session = Depends(get_session)) -> CollectionInfo:
    """Registers a new video library. Fill it with POST /admin/import."""
    if session.get(Collection, req.id) is not None:
        raise ApiError(409, "exists", f"Collection '{req.id}' already exists.")
    session.add(Collection(id=req.id, name=req.name, description=req.description))
    session.commit()
    return next(c for c in catalog.collection_infos(session) if c.id == req.id)


def _import_job(collection: str, folder: str, job_id: str) -> None:
    from app.services.ingest_service import update_job

    def progress(i, n, video_id, skipped):
        update_job(job_id, status="embedding", progress=round(i / max(n, 1), 3), step_detail=f"{i}/{n} {video_id}")

    try:
        count = import_folder(collection, Path(folder), on_progress=progress)
        update_job(job_id, status="indexed", progress=1.0, step_detail=f"Imported {count} video(s)")
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), step_detail="Failed")


@router.post("/admin/import", status_code=202, response_model=JobAccepted, dependencies=[Depends(require_admin)],
             responses={404: {"model": ErrorResponse}})
def import_library(req: ImportRequest, background: BackgroundTasks, session: Session = Depends(get_session)) -> JobAccepted:
    """Indexes every video in a server folder (+ optional metadata.csv) into a collection."""
    catalog.require_collection(session, req.collection)
    folder = Path(req.path)
    if not folder.is_absolute():
        folder = get_settings().seed_dir / folder
    if not folder.is_dir():
        raise ApiError(404, "not_found", f"Folder '{folder}' does not exist on the server.")
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    session.add(Job(id=job_id, kind="import", status="queued", step_detail=f"Import {folder.name}"))
    session.commit()
    background.add_task(_import_job, req.collection, str(folder), job_id)
    return JobAccepted(job_id=job_id)
