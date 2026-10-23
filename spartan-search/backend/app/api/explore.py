import time

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db.session import get_session
from app.db.tables import Feedback, Video, Window
from app.errors import ApiError
from app.schemas.api import (
    AskResponse,
    BenchmarkRow,
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    FeedbackRequest,
    LocalizeRequest,
    LocalizeResponse,
    Span,
)
from app.services import catalog, localize_service
from app.services.benchmarks_service import load_benchmarks
from app.services.compare_service import run_compare
from app.services.localize_service import SALIENCY_FPS

router = APIRouter(tags=["explore"])
ERRORS = {404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}}


def _ground(session: Session, video_id: str, query: str, model: str):
    video = session.get(Video, video_id)
    if video is None:
        raise ApiError(404, "not_found", f"Video '{video_id}' does not exist.")
    adapter = catalog.require_model(model)
    if not adapter.supports_localize:
        raise ApiError(409, "localize_unsupported", f"Model '{model}' cannot answer questions about a video.")
    t0 = time.perf_counter()
    text = adapter.encode_text([query])[0]
    windows = list(session.exec(select(Window).where(Window.video_id == video_id).order_by(Window.start)).all())
    spans, saliency = localize_service.localize(adapter, model, text, windows, video.duration)
    if not windows or (not spans and saliency is None):
        raise ApiError(409, "index_not_ready", f"This video is not indexed for {model} yet.")
    return video, spans, saliency or [], int(round((time.perf_counter() - t0) * 1000))


@router.post("/videos/{video_id}/localize", response_model=LocalizeResponse, responses=ERRORS)
def localize(video_id: str, req: LocalizeRequest, session: Session = Depends(get_session)) -> LocalizeResponse:
    _, spans, saliency, latency = _ground(session, video_id, req.query, req.model)
    return LocalizeResponse(
        video_id=video_id, spans=[Span(start=s.start, end=s.end, score=s.score) for s in spans],
        saliency=saliency, saliency_fps=SALIENCY_FPS, latency_ms=latency,
    )


@router.post("/videos/{video_id}/ask", response_model=AskResponse, responses=ERRORS)
def ask(video_id: str, req: LocalizeRequest, session: Session = Depends(get_session)) -> AskResponse:
    """Answer a question from this video only. The answer is the matching moment."""
    _, spans, saliency, latency = _ground(session, video_id, req.query, req.model)
    return AskResponse(
        video_id=video_id, model=req.model, query=req.query,
        answer=localize_service.grounded_answer(req.query, spans),
        spans=[Span(start=s.start, end=s.end, score=s.score) for s in spans],
        saliency=saliency, saliency_fps=SALIENCY_FPS, latency_ms=latency,
    )


@router.post("/search/compare", response_model=CompareResponse, responses=ERRORS)
def compare(req: CompareRequest, session: Session = Depends(get_session)) -> CompareResponse:
    return run_compare(session, req)


@router.get("/benchmarks", response_model=list[BenchmarkRow])
def benchmarks(dataset: str | None = None, model: str | None = None) -> list[BenchmarkRow]:
    return load_benchmarks(dataset, model)


@router.post("/feedback", status_code=201, responses={404: {"model": ErrorResponse}})
def feedback(req: FeedbackRequest, session: Session = Depends(get_session)) -> dict:
    if session.get(Video, req.video_id) is None:
        raise ApiError(404, "not_found", f"Video '{req.video_id}' does not exist.")
    session.add(Feedback(
        query_id=req.query_id, video_id=req.video_id, relevant=req.relevant,
        span_start=req.span.start if req.span else None, span_end=req.span.end if req.span else None,
    ))
    session.commit()
    return {"status": "recorded"}
