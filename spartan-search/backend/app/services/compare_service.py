import re
import time

from sqlmodel import Session, select

from app.db.tables import Video
from app.schemas.api import CompareRequest, CompareResponse, CompareRun
from app.services import catalog
from app.services.search_service import _Timer, add_spans, rank


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def ground_truth(session: Session, collection: str, query: str) -> str | None:
    """A video counts as ground truth when one of its captions matches the query exactly."""
    target = _norm(query)
    for video in session.exec(select(Video).where(Video.collection == collection, Video.captions != "")).all():
        if any(_norm(c) == target for c in video.captions.split("|")):
            return video.id
    return None


def run_compare(session: Session, req: CompareRequest) -> CompareResponse:
    catalog.require_collection(session, req.collection)
    runs = []
    for model_id in dict.fromkeys(req.models):
        adapter = catalog.require_model(model_id)
        catalog.require_index(model_id, req.collection)
        t0 = time.perf_counter()
        timer = _Timer()
        text = adapter.encode_text([req.query])[0]
        results, window_ids = rank(session, adapter, model_id, req.collection, text, req.top_k, True, None, timer)
        if adapter.supports_localize:
            add_spans(session, adapter, model_id, text, results, window_ids)
        runs.append(CompareRun(model=model_id, results=results,
                               latency_ms=int(round((time.perf_counter() - t0) * 1000))))
    return CompareResponse(query=req.query, runs=runs, ground_truth_video_id=ground_truth(session, req.collection, req.query))
