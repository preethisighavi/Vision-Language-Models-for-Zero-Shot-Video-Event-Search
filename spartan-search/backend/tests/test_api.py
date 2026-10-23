from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg
API = "/api/v1"
ADMIN = {"X-Admin-Key": "test-admin"}


def search(client, **overrides):
    body = {"query": "a dog catches a frisbee", "model": "mock", "collection": "msrvtt-1ka", "top_k": 5}
    body.update(overrides)
    return client.post(f"{API}/search", json=body)


def test_health_models_collections(client):
    health = client.get(f"{API}/health").json()
    assert health["status"] == "ok"
    assert health["index_sizes"]["mock"]["msrvtt-1ka"] == 3

    models = client.get(f"{API}/models").json()
    assert {m["id"] for m in models} == {"mock", "mock-baseline"}
    assert sum(m["default"] for m in models) == 1

    collections = {c["id"]: c for c in client.get(f"{API}/collections").json()}
    assert collections["msrvtt-1ka"]["video_count"] == 3
    assert set(collections["msrvtt-1ka"]["indexed_models"]) == {"mock", "mock-baseline"}
    assert collections["uploads"]["indexed_models"] == []


def test_search_ranks_by_meaning_and_returns_spans(client):
    data = search(client).json()
    assert data["results"][0]["video_id"] == "video9001"
    assert data["total_videos"] == 3
    for r in data["results"]:
        assert 0 <= r["score"] <= 1
        assert r["stream_url"].endswith("/stream")
        assert len(r["spans"]) <= 3
        for s in r["spans"]:
            assert 0 <= s["start"] < s["end"] <= r["duration"] + 1e-6
            assert s["end"] - s["start"] >= 1.0 - 1e-6
    assert set(data["latency_ms"]) == {"encode_text", "stage1", "rerank", "localize", "total"}


def test_rerank_switch(client):
    on = search(client, rerank=True).json()
    off = search(client, rerank=False).json()
    assert all(r["rerank_score"] is not None for r in on["results"])
    assert all(r["rerank_score"] is None for r in off["results"])
    # mock-baseline does not support re-ranking, so the flag is ignored
    base = search(client, model="mock-baseline", rerank=True).json()
    assert all(r["rerank_score"] is None for r in base["results"])


def test_search_errors_use_error_shape(client):
    r = search(client, model="nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_model"
    r = search(client, collection="uploads")
    assert r.status_code == 409 and r.json()["error"]["code"] == "index_not_ready"
    r = search(client, top_k=51)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = search(client, query="x" * 201)
    assert r.status_code == 422


def test_stream_supports_ranges(client):
    full = client.get(f"{API}/videos/video9001/stream")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    part = client.get(f"{API}/videos/video9001/stream", headers={"Range": "bytes=10-109"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 10-109/{len(full.content)}"
    assert part.content == full.content[10:110]
    bad = client.get(f"{API}/videos/video9001/stream", headers={"Range": f"bytes={len(full.content) + 5}-"})
    assert bad.status_code == 416


def test_thumbnail_and_detail(client):
    thumb = client.get(f"{API}/videos/video9001/thumbnail?t=1.5")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    detail = client.get(f"{API}/videos/video9001").json()
    assert detail["windows"] and detail["windows"][0]["start"] == 0.0
    page = client.get(f"{API}/videos?collection=msrvtt-1ka&page_size=2").json()
    assert page["total"] == 3 and len(page["items"]) == 2


def test_ask_is_grounded_in_one_video(client):
    r = client.post("/api/v1/videos/video9001/ask", json={"query": "where is the dog", "model": "mock"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video_id"] == "video9001"
    assert "closest moment" in body["answer"]
    assert body["spans"] and body["model"] == "mock"


def test_localize(client):
    r = client.post(f"{API}/videos/video9002/localize", json={"query": "cooking", "model": "mock"}).json()
    assert r["video_id"] == "video9002" and r["spans"] and r["saliency"]


def test_compare_with_ground_truth(client):
    body = {"query": "a dog catches a frisbee in the park", "models": ["mock", "mock-baseline"], "collection": "msrvtt-1ka", "top_k": 3}
    r = client.post(f"{API}/search/compare", json=body).json()
    assert [run["model"] for run in r["runs"]] == ["mock", "mock-baseline"]
    assert r["ground_truth_video_id"] == "video9001"


def test_benchmarks_long_format(client):
    rows = client.get(f"{API}/benchmarks").json()
    metrics = {row["metric"] for row in rows}
    assert metrics == {"R@1", "latency_ms"}  # Top1Accuracy duplicates R@1 and is dropped
    r1 = next(row for row in rows if row["metric"] == "R@1")
    assert r1["model"] == "CLIP (fine-tuned)" and r1["dataset"] == "MSVD" and r1["protocol"]
    latency = next(row for row in rows if row["metric"] == "latency_ms")
    assert abs(latency["value"] - 36.5) < 1e-6


def test_feedback_and_admin(client):
    assert client.post(f"{API}/feedback", json={"query_id": "q_1", "video_id": "video9001", "relevant": True}).status_code == 201
    assert client.get(f"{API}/admin/stats").status_code == 401
    stats = client.get(f"{API}/admin/stats", headers=ADMIN).json()
    assert stats["queries_total"] >= 1 and "mock" in stats["by_model"]


def test_upload_flow(client, tiny_video):
    with open(tiny_video, "rb") as f:
        r = client.post(f"{API}/videos", files={"file": ("my_birthday.mp4", f, "video/mp4")})
    assert r.status_code == 202
    video_id, job_id = r.json()["video_id"], r.json()["job_id"]
    job = client.get(f"{API}/jobs/{job_id}").json()  # BackgroundTasks run before TestClient returns
    assert job["status"] == "indexed" and job["progress"] == 1.0

    hits = search(client, query="birthday", collection="uploads", video_id=video_id).json()["results"]
    assert [h["video_id"] for h in hits] == [video_id] and hits[0]["spans"]

    assert client.delete(f"{API}/videos/{video_id}").status_code == 401
    assert client.delete(f"{API}/videos/{video_id}", headers=ADMIN).status_code == 204
    assert client.get(f"{API}/videos/{video_id}").status_code == 404


def test_upload_rejects_bad_files(client, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with open(txt, "rb") as f:
        r = client.post(f"{API}/videos", files={"file": ("notes.txt", f, "text/plain")})
    assert r.status_code == 415
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video")
    with open(fake, "rb") as f:
        r = client.post(f"{API}/videos", files={"file": ("fake.mp4", f, "video/mp4")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unreadable_video"
