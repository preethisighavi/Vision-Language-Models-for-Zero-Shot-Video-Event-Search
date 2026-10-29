"""Imports a folder of videos (+ optional metadata.csv) into a collection.

This is the hook for adding real libraries later: drop clips into
DATA_DIR/seed/<collection>/ and run `python -m seed`, or call
POST /admin/import with the folder path.
"""

import csv
import logging
from pathlib import Path

from app.db.session import session_scope
from app.db.tables import Collection, Video
from app.services import ingest_service

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def read_metadata(folder: Path) -> dict[str, dict[str, str]]:
    path = folder / "metadata.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["video_id"]: row for row in csv.DictReader(f) if row.get("video_id")}


def import_folder(collection_id: str, folder: Path, force: bool = False, model_ids: list[str] | None = None,
                  on_progress=None) -> int:
    folder = Path(folder)
    meta = read_metadata(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES) if folder.exists() else []
    with session_scope() as s:
        if s.get(Collection, collection_id) is None:
            s.add(Collection(id=collection_id, name=collection_id, description=""))
            s.commit()

    imported = 0
    for i, path in enumerate(files, start=1):
        video_id = path.stem
        row = meta.get(video_id, {})
        with session_scope() as s:
            existing = s.get(Video, video_id)
            if existing and existing.status == "ready" and not force:
                if on_progress:
                    on_progress(i, len(files), video_id, skipped=True)
                continue
            if existing is None:
                existing = Video(id=video_id, collection=row.get("collection") or collection_id, title=video_id)
            existing.title = row.get("title") or existing.title
            existing.source = row.get("source", existing.source)
            existing.license = row.get("license", existing.license)
            existing.captions = row.get("captions", existing.captions)
            existing.status = "processing"
            s.add(existing)
            s.commit()
        try:
            ingest_service.prepare_media(video_id, path)
            ingest_service.embed_video(video_id, model_ids)
            with session_scope() as s:
                video = s.get(Video, video_id)
                video.status = "ready"
                s.add(video)
                s.commit()
            imported += 1
        except Exception:
            log.exception("import failed for %s", path)
            with session_scope() as s:
                video = s.get(Video, video_id)
                if video:
                    video.status = "failed"
                    s.add(video)
                    s.commit()
        if on_progress:
            on_progress(i, len(files), video_id, skipped=False)
    return imported
