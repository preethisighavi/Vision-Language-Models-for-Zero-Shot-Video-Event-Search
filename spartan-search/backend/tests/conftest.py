import csv
import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="spartan-test-"))
os.environ.update({
    "DATA_DIR": str(_TMP / "data"),
    "MOCK_LATENCY": "false",
    "ADMIN_KEY": "test-admin",
    "REDIS_URL": "",
    "RESULTS_CSV": str(_TMP / "model_results.csv"),
    "SEARCH_RATE_PER_MIN": "100000",
    "UPLOAD_RATE_PER_HOUR": "1000",
})

from fastapi.testclient import TestClient  # noqa: E402

from app.services import media  # noqa: E402

requires_ffmpeg = pytest.mark.skipif(not media.ffmpeg_available(), reason="ffmpeg is not installed")

CLIPS = [
    ("msrvtt-1ka", "video9001", "a dog catches a frisbee in the park"),
    ("msrvtt-1ka", "video9002", "a man is cooking pasta in a kitchen"),
    ("msrvtt-1ka", "video9003", "people playing basketball in a gym"),
    ("msvd-test", "msvd9001", "a woman is slicing an onion"),
]


def reset_caches(models_yaml: Path | None = None) -> None:
    from app.config import get_settings
    from app.db.session import get_engine
    from app.index.factory import reset_index_cache
    from app.models.registry import reset_registry

    if models_yaml is not None:
        os.environ["MODELS_YAML"] = str(models_yaml)
    get_settings.cache_clear()
    get_engine.cache_clear()
    reset_registry()
    reset_index_cache()


def write_results_csv(path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "dataset", "metric_name", "metric_value", "source_file"])
        w.writerow(["CLIP", "fine_tuned", "MSVD", "R@1", "0.366", "clip_finetuned_results.json"])
        w.writerow(["CLIP", "fine_tuned", "MSVD", "Top1Accuracy", "0.366", "clip_finetuned_results.json"])
        w.writerow(["CLIP", "fine_tuned", "MSVD", "avg_query_latency_sec", "0.0365", "clip_finetuned_results.json"])


def seed_clips(seed_root: Path) -> None:
    from app.db.session import init_db, session_scope
    from app.services.catalog import ensure_default_collections
    from app.services.library_service import import_folder

    init_db()
    with session_scope() as s:
        ensure_default_collections(s)
    for collection in {c for c, _, _ in CLIPS}:
        folder = seed_root / collection
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for r in CLIPS if r[0] == collection]
        for _, vid, _title in rows:
            if not (folder / f"{vid}.mp4").exists():
                media.make_test_video(folder / f"{vid}.mp4", vid, 4, "0x0a66e0")
        with open(folder / "metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "title", "collection", "source", "license", "captions"])
            w.writeheader()
            for c, vid, title in rows:
                w.writerow({"video_id": vid, "title": title, "collection": c, "source": "test", "license": "CC0", "captions": title})
        import_folder(collection, folder)


@pytest.fixture(scope="session")
def client():
    if not media.ffmpeg_available():
        pytest.skip("ffmpeg is not installed")
    write_results_csv(_TMP / "model_results.csv")
    reset_caches(Path(__file__).resolve().parent / "models.yaml")
    seed_clips(_TMP / "seed")
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def tiny_video(tmp_path) -> Path:
    path = tmp_path / "clip.mp4"
    media.make_test_video(path, "upload", 6, "0x2e9e4f")
    return path
