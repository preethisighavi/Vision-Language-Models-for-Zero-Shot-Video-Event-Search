"""Text-to-video endpoints, with the GPU generation server mocked."""

from app.services import generation_service, media
from app.services.generation_service import GeneratorError

API = "/api/v1"
MODELS = [{"id": "spartangen-tiny", "label": "SpartanGen-Tiny"}, {"id": "wan-1.3b", "label": "Wan 2.1 1.3B"}]


def _offline():
    raise GeneratorError("The generation server is offline.")


def test_generator_offline(client, monkeypatch):
    monkeypatch.setattr(generation_service, "list_generator_models", _offline)
    status = client.get(f"{API}/generate/models").json()
    assert status["available"] is False and status["models"] == []
    r = client.post(f"{API}/generate", json={"prompt": "a dog on a beach", "model": "wan-1.3b"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "generator_unavailable"


def test_generate_flow_makes_clip_searchable(client, monkeypatch, tmp_path):
    clip = tmp_path / "gen.mp4"
    media.make_test_video(clip, "generated", 3, "0x8a2be2")
    calls = []

    def fake_generate(prompt, model, seed):
        calls.append((prompt, model, seed))
        return clip.read_bytes()

    monkeypatch.setattr(generation_service, "list_generator_models", lambda: MODELS)
    monkeypatch.setattr(generation_service, "call_generator", fake_generate)

    assert [m["id"] for m in client.get(f"{API}/generate/models").json()["models"]] == ["spartangen-tiny", "wan-1.3b"]
    assert client.post(f"{API}/generate", json={"prompt": "a dog", "model": "sora"}).status_code == 400
    assert client.post(f"{API}/generate", json={"prompt": "hi", "model": "wan-1.3b"}).status_code == 422

    r = client.post(f"{API}/generate", json={"prompt": "  a purple   balloon floats  ", "model": "wan-1.3b", "seed": 7})
    assert r.status_code == 202
    video_id, job_id = r.json()["video_id"], r.json()["job_id"]
    assert calls == [("a purple balloon floats", "wan-1.3b", 7)]

    job = client.get(f"{API}/jobs/{job_id}").json()  # BackgroundTasks finish before TestClient returns
    assert job["status"] == "indexed", job
    video = client.get(f"{API}/videos/{video_id}").json()
    assert video["collection"] == "generated" and video["title"] == "a purple balloon floats"

    hits = client.post(f"{API}/search", json={"query": "purple balloon", "model": "mock", "collection": "generated", "top_k": 5}).json()
    assert video_id in [h["video_id"] for h in hits["results"]]


def test_generation_failure_marks_job_failed(client, monkeypatch):
    monkeypatch.setattr(generation_service, "list_generator_models", lambda: MODELS)

    def boom(prompt, model, seed):
        raise GeneratorError("Generator returned 500: CUDA out of memory")

    monkeypatch.setattr(generation_service, "call_generator", boom)
    r = client.post(f"{API}/generate", json={"prompt": "a cat jumps", "model": "spartangen-tiny"})
    job = client.get(f"{API}/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "failed" and "out of memory" in job["error"]
