"""Turns a registered video into playback media, windows, vectors and features."""

import logging
import uuid
from pathlib import Path

from sqlmodel import select

from app.db.session import session_scope
from app.db.tables import Job, Video, Window, utcnow
from app.index.factory import get_index
from app.models.base import WindowContext
from app.models.registry import get_registry
from app.services import media
from app.store.feature_store import FeatureStore
from app.store.media_store import MediaStore

log = logging.getLogger(__name__)

UPLOADS = "uploads"
TRIMMED_CLIP_MAX_SECONDS = 60.0


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def update_job(job_id: str | None, **fields) -> None:
    if not job_id:
        return
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = utcnow()
        s.add(job)
        s.commit()


def _set_video(video_id: str, **fields) -> None:
    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is None:
            return
        for key, value in fields.items():
            setattr(video, key, value)
        s.add(video)
        s.commit()


def _window_settings() -> tuple[float, float]:
    registry = get_registry()
    adapter = registry.peek(registry.default_id) if registry.default_id else None
    if adapter is None:
        return 10.0, 5.0
    return float(adapter.window_seconds), float(adapter.window_stride)


def prepare_media(video_id: str, source: Path, job_id: str | None = None) -> list[Window]:
    """Transcode, probe, make posters and compute windows. Returns the windows."""
    store = MediaStore()
    update_job(job_id, status="decoding", progress=0.15, step_detail="Decoding and transcoding")
    playback = store.video_path(video_id)
    media.transcode_playback(source, playback)
    info = media.probe(playback)

    with session_scope() as s:
        video = s.get(Video, video_id)
        collection = video.collection if video else UPLOADS
        if video:
            video.duration, video.width, video.height = info.duration, info.width, info.height
            video.status = "processing"
            s.add(video)
            s.commit()

    try:
        media.extract_thumbnail(playback, min(info.duration * 0.25, max(info.duration - 0.1, 0)), store.poster_path(video_id))
        media.extract_sprite(playback, info.duration, store.sprite_path(video_id))
    except media.MediaError:
        log.warning("poster/sprite failed for %s", video_id)

    whole_clip = collection != UPLOADS and info.duration <= TRIMMED_CLIP_MAX_SECONDS
    window_seconds, stride = _window_settings()
    spans = media.compute_windows(info.duration, window_seconds, stride, whole_clip)

    with session_scope() as s:
        for old in s.exec(select(Window).where(Window.video_id == video_id)).all():
            s.delete(old)
        s.commit()
        windows = [Window(id=f"{video_id}_w{i:03d}", video_id=video_id, start=a, end=b) for i, (a, b) in enumerate(spans)]
        s.add_all(windows)
        s.commit()
    update_job(job_id, progress=0.3, step_detail=f"{len(windows)} window(s)")
    return windows


def embed_video(video_id: str, model_ids: list[str] | None = None, job_id: str | None = None,
                progress_range: tuple[float, float] = (0.35, 0.95)) -> None:
    """Encodes every window of a video with each model and writes vectors + features."""
    registry = get_registry()
    model_ids = model_ids or registry.enabled_ids()
    store, features = MediaStore(), FeatureStore()
    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is None:
            raise ValueError(f"unknown video {video_id}")
        windows = s.exec(select(Window).where(Window.video_id == video_id).order_by(Window.start)).all()
        collection, title = video.collection, video.title
    playback = store.video_path(video_id)
    update_job(job_id, status="embedding", progress=progress_range[0], step_detail="Encoding windows")

    frame_cache: dict[tuple[str, int], tuple] = {}
    total = max(len(model_ids) * len(windows), 1)
    done = 0
    for model_id in model_ids:
        adapter = registry.get(model_id)
        index = get_index(model_id, collection)
        n = int(adapter.frames_per_window)
        for w in windows:
            key = (w.id, n)
            if key not in frame_cache:
                times = media.segment_times(w.start, w.end, n)
                frame_cache[key] = (media.sample_window_frames(playback, w.start, w.end, n), times)
            frames, times = frame_cache[key]
            enc = adapter.encode_window(
                frames, times, WindowContext(video_id=video_id, window_id=w.id, title=title, start=w.start, end=w.end)
            )
            index.upsert([w.id], enc.embedding[None, :], [{"video_id": video_id, "start": w.start, "end": w.end}])
            features.put(model_id, w.id, enc.frame_features, enc.frame_times)
            done += 1
            lo, hi = progress_range
            update_job(job_id, progress=round(lo + (hi - lo) * done / total, 3),
                       step_detail=f"Encoding with {model_id} ({done}/{total})")


def remove_vectors(video_id: str, model_ids: list[str] | None = None) -> None:
    registry = get_registry()
    with session_scope() as s:
        video = s.get(Video, video_id)
        windows = s.exec(select(Window).where(Window.video_id == video_id)).all()
        collection = video.collection if video else UPLOADS
    ids = [w.id for w in windows]
    for model_id in model_ids or registry.enabled_ids():
        get_index(model_id, collection).delete(ids)
        FeatureStore().delete(model_id, ids)


def ingest_video(video_id: str, source_path: str, job_id: str | None = None) -> None:
    """Full pipeline for a newly registered video. Safe to run in a worker process."""
    try:
        update_job(job_id, status="uploaded", progress=0.1, step_detail="File received")
        prepare_media(video_id, Path(source_path), job_id)
        embed_video(video_id, job_id=job_id)
        _set_video(video_id, status="ready")
        update_job(job_id, status="indexed", progress=1.0, step_detail="Searchable now")
    except Exception as exc:
        log.exception("ingest failed for %s", video_id)
        try:
            remove_vectors(video_id)
        except Exception:
            log.exception("cleanup failed for %s", video_id)
        _set_video(video_id, status="failed")
        message = str(exc) if isinstance(exc, media.MediaError) else "Processing failed. Try another file."
        update_job(job_id, status="failed", error=message, step_detail="Failed")
        raise
    finally:
        upload_copy = Path(source_path)
        if upload_copy.parent.name == "uploads" and upload_copy.exists():
            upload_copy.unlink(missing_ok=True)


def reindex_collection(model_id: str, collection_id: str, job_id: str | None = None) -> None:
    with session_scope() as s:
        ids = [v.id for v in s.exec(
            select(Video).where(Video.collection == collection_id, Video.status == "ready")
        ).all()]
    try:
        update_job(job_id, status="embedding", progress=0.0, step_detail=f"0/{len(ids)} videos")
        for i, video_id in enumerate(ids, start=1):
            embed_video(video_id, [model_id])
            update_job(job_id, progress=round(i / max(len(ids), 1), 3), step_detail=f"{i}/{len(ids)} videos")
        update_job(job_id, status="indexed", progress=1.0, step_detail=f"Reindexed {len(ids)} videos")
    except Exception as exc:
        log.exception("reindex failed")
        update_job(job_id, status="failed", error=str(exc), step_detail="Failed")
        raise
