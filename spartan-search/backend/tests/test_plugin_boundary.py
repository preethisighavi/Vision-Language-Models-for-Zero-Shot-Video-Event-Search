"""A model registered only through models.yaml works end to end, with no changes to app/."""

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import requires_ffmpeg, reset_caches, seed_clips

pytestmark = requires_ffmpeg


def test_dummy_adapter_via_yaml_only():
    tmp = Path(tempfile.mkdtemp(prefix="spartan-plugin-"))
    yaml_path = tmp / "models.yaml"
    yaml_path.write_text(
        "default: dummy\n"
        "models:\n"
        "  - id: dummy\n"
        "    name: Dummy\n"
        "    description: Plugged in from tests/\n"
        "    class: tests.dummy_adapter.DummyAdapter\n"
    )
    old_data_dir = os.environ["DATA_DIR"]
    old_yaml = os.environ.get("MODELS_YAML")
    os.environ["DATA_DIR"] = str(tmp / "data")
    try:
        reset_caches(yaml_path)
        seed_clips(tmp / "seed")
        from app.main import create_app

        with TestClient(create_app()) as c:
            models = c.get("/api/v1/models").json()
            assert [m["id"] for m in models] == ["dummy"]
            assert models[0]["embedding_dim"] == 8 and models[0]["supports_rerank"] is False

            r = c.post("/api/v1/search", json={"query": "anything", "model": "dummy", "collection": "msrvtt-1ka"})
            assert r.status_code == 200, r.text
            results = r.json()["results"]
            assert len(results) == 3
            assert all(res["spans"] and res["rerank_score"] is None for res in results)
    finally:
        os.environ["DATA_DIR"] = old_data_dir
        if old_yaml:
            os.environ["MODELS_YAML"] = old_yaml
        reset_caches(Path(old_yaml) if old_yaml else None)
