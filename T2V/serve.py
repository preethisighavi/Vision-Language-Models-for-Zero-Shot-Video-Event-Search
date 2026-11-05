"""Generation server for the GPU machine. Spartan Search's backend calls it over HTTP.

  T2V_MODELS=spartangen-tiny,wan-1.3b SPARTANGEN_CKPT=T2V/runs/tiny_video/last.pt \
  T2V_API_KEY=change-me uvicorn T2V.serve:app --host 0.0.0.0 --port 8100

GET  /health    -> available models
POST /generate  {"prompt", "model", "seed", "steps"?, "cfg"?} -> video/mp4 bytes,
                generation metadata in the X-Generation-Info header (JSON)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from T2V.generators import REGISTRY, get_generator

ENABLED = [m.strip() for m in os.environ.get("T2V_MODELS", ",".join(REGISTRY)).split(",") if m.strip()]
API_KEY = os.environ.get("T2V_API_KEY", "")
_gpu = threading.Lock()  # one generation at a time on a single GPU

app = FastAPI(title="Spartan Gen server", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    model: str = "spartangen-tiny"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    steps: int | None = Field(default=None, ge=4, le=100)
    cfg: float | None = Field(default=None, ge=1.0, le=15.0)


def _check_key(key: str | None):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.on_event("startup")
def _preload():
    if os.environ.get("T2V_PRELOAD") == "1":
        for m in ENABLED:
            get_generator(m)


@app.get("/health")
def health():
    return {"ok": True, "models": [{"id": m, "label": REGISTRY[m].label} for m in ENABLED if m in REGISTRY]}


@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    if req.model not in ENABLED:
        raise HTTPException(status_code=400, detail=f"model must be one of {ENABLED}")
    with _gpu:
        gen = get_generator(req.model)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.mp4"
            t0 = time.time()
            info = gen.generate(req.prompt, out, seed=req.seed, steps=req.steps, cfg=req.cfg)
            info["seconds"] = round(time.time() - t0, 2)
            data = out.read_bytes()
    return Response(content=data, media_type="video/mp4", headers={"X-Generation-Info": json.dumps(info)})
