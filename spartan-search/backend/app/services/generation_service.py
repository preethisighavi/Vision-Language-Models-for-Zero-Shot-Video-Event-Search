"""Text-to-video: ask the GPU generation server (T2V/serve.py) for a clip, then ingest it.

A generated clip goes through the normal ingest pipeline into the "generated"
collection, so it is searchable straight away with every retrieval model.
"""

import json
import logging
import urllib.error
import urllib.request

from app.config import get_settings
from app.services import ingest_service
from app.services.ingest_service import _set_video, update_job
from app.store.media_store import MediaStore

log = logging.getLogger(__name__)

GENERATED = "generated"


class GeneratorError(RuntimeError):
    pass


def _request(path: str, body: dict | None = None, timeout: float = 5.0) -> tuple[bytes, dict]:
    settings = get_settings()
    if not settings.generator_url:
        raise GeneratorError("Video generation is not configured on this server.")
    url = settings.generator_url.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if settings.generator_api_key:
        headers["X-API-Key"] = settings.generator_api_key
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise GeneratorError(f"Generator returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeneratorError("The generation server is offline.") from exc


def list_generator_models() -> list[dict]:
    """[{id, label}] from the generation server's /health."""
    raw, _ = _request("/health", timeout=3.0)
    return json.loads(raw)["models"]


def call_generator(prompt: str, model: str, seed: int) -> bytes:
    """mp4 bytes for the prompt."""
    raw, _ = _request("/generate", {"prompt": prompt, "model": model, "seed": seed},
                      timeout=get_settings().generator_timeout_seconds)
    return raw


def generate_video(video_id: str, prompt: str, model: str, seed: int, job_id: str) -> None:
    """Worker task: generate -> store -> ingest (windows, vectors, thumbnails)."""
    update_job(job_id, status="generating", progress=0.05, step_detail=f"Generating with {model}")
    try:
        data = call_generator(prompt, model, seed)
        if not data:
            raise GeneratorError("The generator returned an empty file.")
    except Exception as exc:
        log.exception("generation failed for %s", video_id)
        _set_video(video_id, status="failed")
        message = str(exc) if isinstance(exc, GeneratorError) else "Generation failed."
        update_job(job_id, status="failed", error=message, step_detail="Failed")
        return
    dest = MediaStore().upload_path(video_id, ".mp4")
    dest.write_bytes(data)
    try:
        ingest_service.ingest_video(video_id, str(dest), job_id)
    except Exception:
        pass  # ingest_video already marked the job and video failed and logged the cause
