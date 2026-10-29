import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from sqlmodel import Session

from app.config import get_settings
from app.db.session import get_session
from app.db.tables import Job, Video
from app.errors import ApiError
from app.schemas.api import ErrorResponse, JobInfo, UploadAccepted
from app.security import client_ip, require_upload_key, upload_limiter
from app.services import media
from app.services.ingest_service import UPLOADS
from app.store.media_store import MediaStore
from app.worker import tasks
from app.worker.queue import enqueue

router = APIRouter(tags=["uploads"])

ALLOWED = {
    ".mp4": {"video/mp4", "application/octet-stream"},
    ".mov": {"video/quicktime", "application/octet-stream"},
    ".webm": {"video/webm", "application/octet-stream"},
}
CHUNK = 1024 * 1024


def _clean_title(raw: str) -> str:
    title = re.sub(r"[_\-]+", " ", Path(raw).stem)
    title = re.sub(r"[^\w\s.,'()&]", "", title).strip()
    return (title or "Untitled video")[:120]


@router.post(
    "/videos",
    status_code=202,
    response_model=UploadAccepted,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    dependencies=[Depends(require_upload_key)],
)
async def upload_video(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    session: Session = Depends(get_session),
) -> UploadAccepted:
    settings = get_settings()
    upload_limiter.check(client_ip(request))

    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if suffix not in ALLOWED or content_type not in ALLOWED[suffix]:
        raise ApiError(415, "unsupported_media", "Upload an MP4, MOV or WebM video.")

    video_id = f"up_{uuid.uuid4().hex[:12]}"
    store = MediaStore()
    dest = store.upload_path(video_id, suffix)
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise ApiError(413, "file_too_large", f"Videos must be {settings.max_upload_mb} MB or smaller.")
            out.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise ApiError(400, "empty_file", "The file is empty.")

    try:
        info = media.probe(dest)
    except media.MediaError:
        dest.unlink(missing_ok=True)
        raise ApiError(415, "unreadable_video", "That file could not be read as a video.")
    if not info.has_video or info.duration <= 0:
        dest.unlink(missing_ok=True)
        raise ApiError(415, "unreadable_video", "That file has no video stream.")
    if info.duration > settings.max_upload_seconds:
        dest.unlink(missing_ok=True)
        minutes = settings.max_upload_seconds // 60
        raise ApiError(413, "video_too_long", f"Videos must be {minutes} minutes or shorter.")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    session.add(Video(
        id=video_id, collection=UPLOADS, title=(title or "").strip()[:120] or _clean_title(file.filename or ""),
        duration=info.duration, width=info.width, height=info.height, source="User upload",
        license="Uploaded by a site visitor", status="queued",
    ))
    session.add(Job(id=job_id, video_id=video_id, kind="ingest", status="queued", progress=0.05,
                    step_detail="Waiting for a worker"))
    session.commit()
    enqueue(background, tasks.ingest_video, video_id, str(dest), job_id)
    return UploadAccepted(video_id=video_id, job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobInfo, responses={404: {"model": ErrorResponse}})
def job_status(job_id: str, session: Session = Depends(get_session)) -> JobInfo:
    job = session.get(Job, job_id)
    if job is None:
        raise ApiError(404, "not_found", f"Job '{job_id}' does not exist.")
    return JobInfo(id=job.id, video_id=job.video_id, status=job.status, progress=job.progress,
                   step_detail=job.step_detail, error=job.error, updated_at=job.updated_at)
