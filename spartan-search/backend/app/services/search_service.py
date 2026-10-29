import json
import logging
import time
import uuid

from sqlmodel import Session, func, select

from app.db.tables import Query, Video
from app.index.factory import get_index
from app.models.base import Candidate, ModelAdapter, TextEncoding
from app.schemas.api import Latency, SearchRequest, SearchResponse, SearchResult, Span
from app.services import catalog, localize_service
from app.services.localize_service import SALIENCY_FPS
from app.store.feature_store import FeatureStore

log = logging.getLogger("spartan.search")

DEFAULT_RERANK_K = 20


class _Timer:
    def __init__(self):
        self.ms: dict[str, int] = {"encode_text": 0, "stage1": 0, "rerank": 0, "localize": 0}
        self._t0 = time.perf_counter()

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.ms[name] += int(round((time.perf_counter() - self.t) * 1000))

        return _Ctx()

    def total(self) -> int:
        return int(round((time.perf_counter() - self._t0) * 1000))


def thumbnail_url(video_id: str, t: float) -> str:
    return f"/api/v1/videos/{video_id}/thumbnail?t={max(t, 0):.1f}"


def stream_url(video_id: str) -> str:
    return f"/api/v1/videos/{video_id}/stream"


def run_search(session: Session, req: SearchRequest, log_query: bool = True) -> SearchResponse:
    adapter = catalog.require_model(req.model)
    catalog.require_collection(session, req.collection)
    catalog.require_index(req.model, req.collection)
    timer = _Timer()

    with timer.stage("encode_text"):
        text = adapter.encode_text([req.query])[0]
    results, window_ids = rank(session, adapter, req.model, req.collection, text, req.top_k, req.rerank,
                               req.video_id, timer)
    if req.with_spans and adapter.supports_localize:
        with timer.stage("localize"):
            add_spans(session, adapter, req.model, text, results, window_ids)

    total_videos = session.exec(
        select(func.count(Video.id)).where(Video.collection == req.collection, Video.status == "ready")
    ).one()
    query_id = f"q_{uuid.uuid4().hex[:10]}"
    latency = Latency(**timer.ms, total=timer.total())
    response = SearchResponse(
        query_id=query_id, query=req.query, model=req.model, collection=req.collection,
        total_videos=int(total_videos), results=results, latency_ms=latency,
    )
    if log_query:
        record_query(session, response, req)
    return response


def rank(session: Session, adapter: ModelAdapter, model_id: str, collection: str, text: TextEncoding,
         top_k: int, rerank: bool, video_id: str | None, timer: _Timer) -> tuple[list[SearchResult], list[str]]:
    """Stage 1 + optional re-rank. Returns results and the best window id of each."""
    index = get_index(model_id, collection)
    with timer.stage("stage1"):
        hits = index.search(text.embedding, k=max(50, top_k * 5), filter={"video_id": video_id} if video_id else None)
        best: dict[str, tuple[str, float, dict]] = {}
        for window_id, score, meta in hits:  # hits are sorted, so the first window per video is its best
            best.setdefault(meta["video_id"], (window_id, score, meta))
        grouped = list(best.values())

    use_rerank = rerank and adapter.supports_rerank
    pool_size = max(int(getattr(adapter, "rerank_k", DEFAULT_RERANK_K)), top_k) if use_rerank else top_k
    pool = grouped[:pool_size]
    rerank_scores: list[float | None] = [None] * len(pool)
    if use_rerank and pool:
        with timer.stage("rerank"):
            feats = FeatureStore().get_many(model_id, [w for w, _, _ in pool])
            candidates = []
            for window_id, score, meta in pool:
                f, t = feats.get(window_id, (None, None))
                candidates.append(Candidate(
                    window_id=window_id, video_id=meta["video_id"], start=float(meta["start"]),
                    end=float(meta["end"]), stage1_score=score, frame_features=f, frame_times=t,
                ))
            rerank_scores = [float(s) for s in adapter.rerank(text, candidates)]

    scored = []
    for (window_id, s1, meta), r in zip(pool, rerank_scores):
        scored.append((adapter.fuse(s1, r), window_id, s1, r, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]

    videos = {v.id: v for v in session.exec(select(Video).where(Video.id.in_([m["video_id"] for *_, m in scored]))).all()}
    results, window_ids = [], []
    for fused, window_id, s1, r, meta in scored:
        video = videos.get(meta["video_id"])
        if video is None:
            continue
        results.append(SearchResult(
            video_id=video.id, title=video.title, collection=video.collection, rank=len(results) + 1,
            score=round(min(max(fused, 0.0), 1.0), 4), stage1_score=round(s1, 4),
            rerank_score=None if r is None else round(r, 4), duration=round(video.duration, 2),
            thumbnail_url=thumbnail_url(video.id, float(meta["start"])), stream_url=stream_url(video.id),
            spans=[], saliency=None, saliency_fps=None,
        ))
        window_ids.append(window_id)
    return results, window_ids


def add_spans(session: Session, adapter: ModelAdapter, model_id: str, text: TextEncoding,
              results: list[SearchResult], window_ids: list[str]) -> None:
    for r, window_id in zip(results, window_ids):
        windows = localize_service.neighbour_windows(session, r.video_id, window_id)
        spans, saliency = localize_service.localize(adapter, model_id, text, windows, r.duration)
        r.spans = [Span(start=s.start, end=s.end, score=s.score) for s in spans]
        r.saliency = saliency
        r.saliency_fps = SALIENCY_FPS if saliency is not None else None
        if r.spans:
            r.thumbnail_url = thumbnail_url(r.video_id, r.spans[0].start)


def record_query(session: Session, response: SearchResponse, req: SearchRequest) -> None:
    session.add(Query(
        id=response.query_id, query=req.query, model=req.model, collection=req.collection, top_k=req.top_k,
        rerank=req.rerank, result_ids_json=json.dumps([r.video_id for r in response.results]),
        latency_json=response.latency_ms.model_dump_json(),
    ))
    session.commit()
    log.info("search", extra={"fields": {
        "query_id": response.query_id, "model": req.model, "collection": req.collection,
        "results": [r.video_id for r in response.results], "latency_ms": response.latency_ms.model_dump(),
    }})

