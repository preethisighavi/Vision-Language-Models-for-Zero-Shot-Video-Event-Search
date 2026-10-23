import re

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, func, select

from app.db.session import get_session
from app.db.tables import Video, Window
from app.errors import ApiError
from app.schemas.api import ErrorResponse, VideoDetail, VideoPage, VideoSummary, WindowInfo
from app.services import media
from app.services.search_service import thumbnail_url
from app.store.media_store import MediaStore

router = APIRouter(tags=["videos"])

ERRORS = {404: {"model": ErrorResponse}}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def get_video_or_404(session: Session, video_id: str) -> Video:
    video = session.get(Video, video_id)
    if video is None:
        raise ApiError(404, "not_found", f"Video '{video_id}' does not exist.")
    return video


@router.get("/videos", response_model=VideoPage)
def list_videos(
    collection: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    session: Session = Depends(get_session),
) -> VideoPage:
    stmt = select(Video)
    count_stmt = select(func.count(Video.id))
    if collection:
        stmt = stmt.where(Video.collection == collection)
        count_stmt = count_stmt.where(Video.collection == collection)
    total = session.exec(count_stmt).one()
    rows = session.exec(
        stmt.order_by(Video.created_at.desc(), Video.id).offset((page - 1) * page_size).limit(page_size)
    ).all()
    items = [
        VideoSummary(
            id=v.id, title=v.title, collection=v.collection, duration=round(v.duration, 2), status=v.status,
            thumbnail_url=thumbnail_url(v.id, min(v.duration * 0.25, 2.0)), created_at=v.created_at,
        )
        for v in rows
    ]
    return VideoPage(items=items, page=page, page_size=page_size, total=int(total))


@router.get("/videos/{video_id}", response_model=VideoDetail, responses=ERRORS)
def video_detail(video_id: str, session: Session = Depends(get_session)) -> VideoDetail:
    v = get_video_or_404(session, video_id)
    windows = session.exec(select(Window).where(Window.video_id == video_id).order_by(Window.start)).all()
    return VideoDetail(
        id=v.id, title=v.title, collection=v.collection, duration=round(v.duration, 2), width=v.width,
        height=v.height, source=v.source, license=v.license, status=v.status, created_at=v.created_at,
        windows=[WindowInfo(id=w.id, start=w.start, end=w.end) for w in windows],
    )


@router.get(
    "/videos/{video_id}/stream",
    responses={206: {"content": {"video/mp4": {}}}, 404: {"model": ErrorResponse}, 416: {"model": ErrorResponse}},
    response_class=StreamingResponse,
)
def stream(video_id: str, range: str | None = Header(default=None), session: Session = Depends(get_session)):
    get_video_or_404(session, video_id)
    store = MediaStore()
    if not store.has_video(video_id):
        raise ApiError(404, "not_found", "The playback file is not ready yet.")
    size = store.size(video_id)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
    start, end = 0, size - 1
    status = 200
    if range:
        match = _RANGE.fullmatch(range.strip())
        if not match or (not match.group(1) and not match.group(2)):
            raise ApiError(416, "bad_range", "Unsupported Range header.")
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
        else:  # suffix range: last N bytes
            start = max(size - int(match.group(2)), 0)
        end = min(end, size - 1)
        if start > end or start >= size:
            raise ApiError(416, "bad_range", "Requested range is not satisfiable.")
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(store.open_range(video_id, start, end), status_code=status,
                             media_type="video/mp4", headers=headers)


@router.get(
    "/videos/{video_id}/thumbnail",
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"model": ErrorResponse}},
    response_class=FileResponse,
)
def thumbnail(video_id: str, t: float = Query(0.0, ge=0), session: Session = Depends(get_session)):
    video = get_video_or_404(session, video_id)
    store = MediaStore()
    if not store.has_video(video_id):
        raise ApiError(404, "not_found", "The video is still processing.")
    t = round(min(t, max(video.duration - 0.05, 0.0)), 1)
    path = store.thumbnail_path(video_id, t)
    if not path.exists():
        try:
            media.extract_thumbnail(store.video_path(video_id), t, path)
        except media.MediaError:
            poster = store.poster_path(video_id)
            if not poster.exists():
                raise ApiError(404, "not_found", "No frame available at that time.")
            path = poster
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
