from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Collection(SQLModel, table=True):
    __tablename__ = "collections"
    id: str = Field(primary_key=True)
    name: str
    description: str = ""


class Video(SQLModel, table=True):
    __tablename__ = "videos"
    id: str = Field(primary_key=True)
    collection: str = Field(index=True, foreign_key="collections.id")
    title: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    source: str = ""
    license: str = ""
    captions: str = ""  # "|"-separated ground-truth captions, used by Compare
    status: str = "queued"  # queued | processing | ready | failed
    owner_session: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class Window(SQLModel, table=True):
    __tablename__ = "windows"
    id: str = Field(primary_key=True)
    video_id: str = Field(index=True, foreign_key="videos.id")
    start: float
    end: float


class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    id: str = Field(primary_key=True)
    video_id: Optional[str] = Field(default=None, index=True)
    kind: str = "ingest"  # ingest | reindex
    status: str = "queued"
    progress: float = 0.0
    step_detail: str = ""
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Query(SQLModel, table=True):
    __tablename__ = "queries"
    id: str = Field(primary_key=True)
    query: str
    model: str = Field(index=True)
    collection: str
    top_k: int
    rerank: bool
    result_ids_json: str = "[]"
    latency_json: str = "{}"
    created_at: datetime = Field(default_factory=utcnow)


class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"
    id: Optional[int] = Field(default=None, primary_key=True)
    query_id: str = Field(index=True)
    video_id: str
    span_start: Optional[float] = None
    span_end: Optional[float] = None
    relevant: bool
    created_at: datetime = Field(default_factory=utcnow)
