"""Video read/write with PyAV. Frames are uint8 numpy arrays shaped (T, H, W, 3)."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def read_frames(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode every frame. Returns (frames[T,H,W,3] uint8, timestamps_seconds[T])."""
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            frames.append(frame.to_ndarray(format="rgb24"))
            if frame.pts is not None and frame.time_base is not None:
                times.append(float(frame.pts * frame.time_base))
            else:
                rate = float(stream.average_rate or 25)
                times.append(i / rate)
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    times_arr = np.asarray(times, dtype=np.float64)
    times_arr -= times_arr[0]
    return np.stack(frames), times_arr


def sample_clip(frames: np.ndarray, times: np.ndarray, num_frames: int, fps: float) -> np.ndarray:
    """Pick `num_frames` frames at `fps` from the middle of the video.

    Videos shorter than the window are sampled uniformly over their full length,
    so every clip yields exactly `num_frames` frames.
    """
    window = (num_frames - 1) / fps
    duration = float(times[-1])
    if duration >= window:
        start = (duration - window) / 2
        targets = start + np.arange(num_frames) / fps
    else:
        targets = np.linspace(0.0, duration, num_frames)
    idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
    return frames[idx]


def uniform_frames(frames: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def write_mp4(frames: np.ndarray, path: str | Path, fps: float = 8.0) -> Path:
    """Encode uint8 frames (T,H,W,3) as browser-playable H.264 mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    # yuv420p needs even dimensions
    h2, w2 = h - h % 2, w - w % 2
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width, stream.height = w2, h2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "movflags": "+faststart"}
        for f in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h2, :w2]), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def to_uint8(video: "np.ndarray") -> np.ndarray:
    """[-1,1] float (T,H,W,3) -> uint8."""
    return ((np.clip(video, -1, 1) + 1) * 127.5).round().astype(np.uint8)
