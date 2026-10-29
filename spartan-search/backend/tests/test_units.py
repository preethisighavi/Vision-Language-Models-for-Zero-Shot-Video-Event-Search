import numpy as np

from app.index.numpy_index import NumpyIndex
from app.models.base import SpanPrediction
from app.models.mock import MockAdapter
from app.services.localize_service import clean_spans, saliency_strip
from app.services.media import compute_windows, segment_times


def test_clean_spans_nms_merge_min_length_and_cap():
    spans = [
        SpanPrediction(2.0, 6.0, 0.9),
        SpanPrediction(2.5, 6.5, 0.8),  # IoU > 0.5 with the first: suppressed
        SpanPrediction(5.5, 8.0, 0.7),  # overlaps the first: merged into it
        SpanPrediction(10.0, 10.2, 0.6),  # too short: widened to 1 s
        SpanPrediction(12.0, 13.0, 0.5),
        SpanPrediction(14.0, 15.0, 0.4),  # 4th span: dropped by the cap
    ]
    out = clean_spans(spans, duration=20.0)
    assert len(out) == 3
    assert (out[0].start, out[0].end, out[0].score) == (2.0, 8.0, 0.9)
    assert abs((out[1].end - out[1].start) - 1.0) < 1e-6
    assert all(a.score >= b.score for a, b in zip(out, out[1:]))


def test_clean_spans_clips_to_duration():
    out = clean_spans([SpanPrediction(-3.0, 50.0, 0.5)], duration=12.0)
    assert (out[0].start, out[0].end) == (0.0, 12.0)


def test_windows_and_segment_sampling():
    assert compute_windows(8.0, 10.0, 5.0, whole_clip=False) == [(0.0, 8.0)]
    assert compute_windows(45.0, 10.0, 5.0, whole_clip=True) == [(0.0, 45.0)]
    windows = compute_windows(27.0, 10.0, 5.0, whole_clip=False)
    assert windows[0] == (0.0, 10.0) and windows[-1][1] == 27.0
    assert all(b - a <= 10.0 for a, b in windows)
    np.testing.assert_allclose(segment_times(0, 10, 5), [1, 3, 5, 7, 9])


def test_saliency_strip_is_per_second():
    strip = saliency_strip(np.array([0.2, 0.9, 0.2], np.float32), np.array([1.0, 3.0, 5.0], np.float32), duration=6.0)
    assert len(strip) == 6 and max(strip) <= 1.0 and strip[2] > strip[0]


def test_numpy_index_roundtrip(tmp_path):
    index = NumpyIndex(tmp_path / "idx")
    vecs = np.eye(3, dtype=np.float32)
    index.upsert(["a", "b", "c"], vecs, [{"video_id": "v1"}, {"video_id": "v2"}, {"video_id": "v2"}])
    assert index.count() == 3
    assert index.search(np.array([0, 1, 0], np.float32), k=1)[0][0] == "b"
    filtered = index.search(np.array([1, 0, 0], np.float32), k=3, filter={"video_id": "v2"})
    assert {h[0] for h in filtered} == {"b", "c"}
    index.upsert(["a"], np.array([[0, 0, 1]], np.float32), [{"video_id": "v1"}])
    assert index.count() == 3 and index.search(np.array([0, 0, 1], np.float32), k=1)[0][0] in {"a", "c"}
    index.delete(["a", "b"])
    assert NumpyIndex(tmp_path / "idx").count() == 1  # persisted


def test_mock_is_deterministic_and_word_aware():
    m = MockAdapter(latency=False)
    a = m.encode_text(["a man is cooking pasta"])[0].embedding
    b = m.encode_text(["a man is cooking pasta"])[0].embedding
    near = m.encode_text(["man cooking"])[0].embedding
    far = m.encode_text(["a cat sleeps on a sofa"])[0].embedding
    np.testing.assert_array_equal(a, b)
    assert float(a @ near) > float(a @ far) + 0.3
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert 0.0 <= m.fuse(0.9, None) <= 1.0
