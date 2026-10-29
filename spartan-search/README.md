# Spartan Search

Type what happens in a video ("a man is cooking pasta") and get a ranked list of videos, each opened at the seconds that match. Users can upload their own videos, compare two models side by side, and see benchmark results.

DATA 298B, Team 1, San José State University. Linear: 298-20 (Website), 298-46 (Search API), 298-47 (Search website). 298-49 (Ensemble ranker) plugs in through `ModelAdapter.fuse` / a new adapter; 298-48 and 298-50 (deploy) are not done yet.

Everything runs end to end today on a **mock model** with deterministic fake results. Real models (CLIP, Spartan VLM) plug in later without touching the frontend or the API.

## Quick start

**Docker** (API, worker, Redis, frontend):

```bash
cd spartan-search
cp .env.example .env        # optional
docker compose up --build
```

Open http://localhost:5173. On first boot the API generates 30 short synthetic clips and indexes them, so the site works with zero downloads.

**Without Docker** (needs Python 3.11, Node 20+, and ffmpeg):

```bash
# backend
cd spartan-search/backend
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m seed                       # generates + indexes the sample clips
.venv/bin/uvicorn app.main:app --port 8000     # API docs at http://localhost:8000/docs

# frontend (second terminal)
cd spartan-search/frontend
npm install
npm run dev                                    # http://localhost:5173, proxies /api to :8000
```

Without `REDIS_URL`, uploads are processed with FastAPI `BackgroundTasks` inside the API process.

**Tests**

```bash
cd backend && .venv/bin/python -m pytest -q    # 19 tests, incl. the plug-in boundary test
cd frontend && npm test && npm run typecheck   # 12 tests
```

## Layout

```
spartan-search/
├─ backend/
│  ├─ app/
│  │  ├─ api/        catalog (health, models, collections), search, videos (stream, thumbnail),
│  │  │              uploads (POST /videos, jobs), explore (localize, compare, benchmarks, feedback), admin
│  │  ├─ models/     ← the model layer: base.py (ModelAdapter), registry.py, mock.py, clip_stub.py, spartan_stub.py
│  │  ├─ index/      VectorIndex protocol, numpy_index.py (default), chroma/pinecone stubs
│  │  ├─ store/      feature_store.py (per-window frame features), media_store.py (videos, thumbnails)
│  │  ├─ services/   search, localize, ingest, compare, benchmarks, library import, ffmpeg helpers
│  │  ├─ db/         SQLModel tables (Postgres-compatible), session
│  │  └─ worker/     RQ tasks + queue fallback
│  ├─ models.yaml    model registry
│  ├─ benchmarks_meta.yaml   protocol/notes for rows in results/model_results.csv
│  ├─ seed/          python -m seed
│  └─ tests/
├─ frontend/         React 18 + TypeScript + Vite + Tailwind + TanStack Query
│  └─ src/{app,pages,components,hooks,api,lib,styles}
├─ docker-compose.yml
└─ .env.example
```

## How to plug in a model

The frontend never knows which model runs, and the API only talks to models through `ModelAdapter` (`backend/app/models/base.py`). The platform does decoding, frame sampling, windowing, indexing, grouping and logging. An adapter only does the maths:

| Method | Called when | Returns |
| --- | --- | --- |
| `load()` | first use (or startup if `PRELOAD_MODELS=true`) | – |
| `encode_text(texts)` | every search | `TextEncoding`: 512-d L2-normalised embedding (+ optional token states) |
| `encode_window(frames, times, context=None)` | indexing; `frames` is `(T, 224, 224, 3)` uint8 RGB | `WindowEncoding`: embedding for the index + per-frame features |
| `rerank(text, candidates)` | search, if `supports_rerank` | one score per candidate |
| `localize(text, frame_features, frame_times)` | search / video detail, if `supports_localize` | spans + per-frame saliency |
| `fuse(stage1, rerank)` | every result | 0–1 display score |

Steps:

1. Implement the class, for example by filling in the `TODO(team)` parts of `clip_stub.py` or `spartan_stub.py`. Any import path works; `tests/dummy_adapter.py` lives outside `app/`.
2. Add or enable its entry in `backend/models.yaml`. `params` are passed to the constructor; `${CHECKPOINT_DIR}` is expanded from the environment. Put checkpoints in `CHECKPOINT_DIR`.
3. Build its index for each library:
   `curl -X POST localhost:8000/api/v1/admin/reindex -H 'X-Admin-Key: change-me' -H 'Content-Type: application/json' -d '{"model":"spartan-v1","collection":"msrvtt-1ka"}'`

Each `(model, collection)` gets its own index, so models never mix. Until a model's index is built, searching it returns `409 index_not_ready`, and the UI shows a banner.

**Score fusion / ensemble (298-49):** an ensemble can be written as one more adapter that wraps several others and combines their scores in `rerank` / `fuse`. No other code changes are needed.

## Adding a real video library

Video libraries are just collections. Two ways to fill one:

- **Seed folder:** put clips in `DATA_DIR/seed/<collection-id>/`, optionally with a `metadata.csv` (`video_id, title, collection, source, license, captions`), and run `python -m seed`. Clips that are already indexed are skipped; use `--force` to redo them.
- **Admin API:** `POST /api/v1/admin/collections` `{id, name, description}` creates a collection, then `POST /api/v1/admin/import` `{collection, path}` indexes a folder on the server in the background. Poll `GET /api/v1/jobs/{job_id}`.

`captions` (separated by `|`) are optional ground truth. When a Compare query exactly matches a caption, that video gets the "Correct" badge.

## API

All endpoints are under `/api/v1`. The OpenAPI spec is at `/openapi.json`, and the frontend types in `frontend/src/api/schema.d.ts` are generated from it with `npm run gen:api`. Errors are always `{"error": {"code", "message"}}`.

`GET /health` · `GET /models` · `GET /collections` · `POST /search` · `POST /search/compare` · `GET /videos` · `GET /videos/{id}` · `GET /videos/{id}/stream` (Range, 206) · `GET /videos/{id}/thumbnail?t=` · `POST /videos/{id}/localize` · `POST /videos` (upload, 202) · `GET /jobs/{id}` · `DELETE /videos/{id}` · `GET /benchmarks` · `POST /feedback` · `POST /admin/reindex` · `GET /admin/stats` · `POST /admin/collections` · `POST /admin/import`

Admin endpoints need `X-Admin-Key`. Uploads need `X-Upload-Key` if `DEMO_UPLOAD_KEY` is set. Search is limited to 60 requests per minute per IP, and uploads to 10 per hour per IP. `SITE_PASSWORD` turns on basic auth for the whole API.

## Decisions where the spec left room

- **`encode_window` takes an optional `context`** (video id, window id, title). Real models should ignore it; the mock needs titles to produce sensible orderings without reading pixels.
- **Two mock models are enabled** (`mock`, `mock-baseline`) so that Compare has two columns before any real model exists. CLIP and Spartan are present in `models.yaml` but disabled.
- **Mock latency:** handlers are synchronous (FastAPI runs them in a thread pool), so the mock uses `time.sleep`, not `asyncio.sleep`. It adds 5–30 ms for text encoding and re-ranking, and 0.5–3 ms per localized video, which keeps `/search` p95 under 100 ms (measured at about 90 ms).
- **Benchmarks:** `results/model_results.csv` has no `protocol`/`notes` columns yet, so they come from `benchmarks_meta.yaml`, or from those columns if they are added later. Top-1/Top-5 accuracy duplicate R@1/R@5 and are dropped; `avg_query_latency_sec` is shown as `latency_ms`.
- **Upload windows** use the default model's `window_seconds`/`window_stride`. Library clips of 60 s or less are one window.
- **Saliency** is resampled to 1 value per second over the whole video.
- **Rate limits** are kept in memory. That is fine for one API process; move them to Redis if the API is scaled out.
- **Uploads are shared**: they are visible to everyone in the `uploads` collection. `videos.owner_session` exists for private uploads later.

## Not done yet

- Real CLIP and Spartan adapters (stubs with TODOs are in place), and the Chroma and Pinecone index backends (stubs).
- Deployment (298-48 API, 298-50 website). The API image is CPU-only and has a health check; the frontend has a static `prod` build target (`docker build --target prod`).
- Dark mode is in the nav (moon / sun). It follows the system setting until you pick one, then remembers the choice.
- Lighthouse has not been run yet.
