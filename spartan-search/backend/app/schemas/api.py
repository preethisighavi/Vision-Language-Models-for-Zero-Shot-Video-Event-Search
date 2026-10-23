from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: str
    models_loaded: list[str]
    index_sizes: dict[str, dict[str, int]]


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    embedding_dim: int
    supports_rerank: bool
    supports_localize: bool
    default: bool
    status: Literal["ready", "loading", "error"]


class CollectionInfo(BaseModel):
    id: str
    name: str
    description: str
    video_count: int
    indexed_models: list[str]


class Span(BaseModel):
    start: float
    end: float
    score: float


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    model: str
    collection: str
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = True
    with_spans: bool = True
    video_id: Optional[str] = None


class SearchResult(BaseModel):
    video_id: str
    title: str
    collection: str
    rank: int
    score: float
    stage1_score: float
    rerank_score: Optional[float] = None
    duration: float
    thumbnail_url: str
    stream_url: str
    spans: list[Span]
    saliency: Optional[list[float]] = None
    saliency_fps: Optional[float] = None


class Latency(BaseModel):
    encode_text: int
    stage1: int
    rerank: int
    localize: int
    total: int


class SearchResponse(BaseModel):
    query_id: str
    query: str
    model: str
    collection: str
    total_videos: int
    results: list[SearchResult]
    latency_ms: Latency


class WindowInfo(BaseModel):
    id: str
    start: float
    end: float


class VideoDetail(BaseModel):
    id: str
    title: str
    collection: str
    duration: float
    width: int
    height: int
    source: str
    license: str
    status: str
    created_at: datetime
    windows: list[WindowInfo]


class VideoSummary(BaseModel):
    id: str
    title: str
    collection: str
    duration: float
    status: str
    thumbnail_url: str
    created_at: datetime


class VideoPage(BaseModel):
    items: list[VideoSummary]
    page: int
    page_size: int
    total: int


class LocalizeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    model: str


class LocalizeResponse(BaseModel):
    video_id: str
    spans: list[Span]
    saliency: list[float]
    saliency_fps: float
    latency_ms: int


class AskResponse(BaseModel):
    video_id: str
    model: str
    query: str
    answer: str
    spans: list[Span]
    saliency: list[float]
    saliency_fps: float
    latency_ms: int


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    models: list[str] = Field(min_length=2, max_length=3)
    collection: str
    top_k: int = Field(default=5, ge=1, le=20)


class CompareRun(BaseModel):
    model: str
    results: list[SearchResult]
    latency_ms: int


class CompareResponse(BaseModel):
    query: str
    runs: list[CompareRun]
    ground_truth_video_id: Optional[str] = None


class UploadAccepted(BaseModel):
    video_id: str
    job_id: str


JobStatus = Literal["queued", "uploaded", "decoding", "embedding", "indexed", "failed"]


class JobInfo(BaseModel):
    id: str
    video_id: Optional[str]
    status: JobStatus
    progress: float
    step_detail: str
    error: Optional[str] = None
    updated_at: datetime


class BenchmarkRow(BaseModel):
    model: str
    dataset: str
    protocol: str
    metric: str
    value: float
    notes: str


class FeedbackRequest(BaseModel):
    query_id: str
    video_id: str
    span: Optional[Span] = None
    relevant: bool


class ReindexRequest(BaseModel):
    model: str
    collection: str


class JobAccepted(BaseModel):
    job_id: str


class CollectionCreate(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,40}$")
    name: str = Field(min_length=1, max_length=80)
    description: str = ""


class ImportRequest(BaseModel):
    collection: str
    path: str = Field(description="Folder on the server, relative to DATA_DIR/seed or absolute")


class Stats(BaseModel):
    queries_total: int
    p50_ms: float
    p95_ms: float
    by_model: dict[str, dict[str, float]]
