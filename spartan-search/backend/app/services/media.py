"""ffmpeg / ffprobe helpers: probing, transcoding, thumbnails, frame sampling."""

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_SIZE = 224


class MediaError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    duration: float
    width: int
    height: int
    has_video: bool


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(args: list[str], timeout: int = 600) -> bytes:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise MediaError("ffmpeg is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("video processing timed out") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or ["unknown error"]
        raise MediaError(tail[0])
    return proc.stdout


def probe(path: Path) -> ProbeResult:
    out = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=60,
    )
    info = json.loads(out or b"{}")
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    duration = float(info.get("format", {}).get("duration") or (video or {}).get("duration") or 0.0)
    return ProbeResult(
        duration=duration,
        width=int((video or {}).get("width") or 0),
        height=int((video or {}).get("height") or 0),
        has_video=video is not None,
    )


def transcode_playback(src: Path, dest: Path) -> None:
    """H.264 MP4 at 480p with faststart, so the browser can seek immediately."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", "scale=-2:'min(480,ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp),
    ])
    tmp.replace(dest)


def extract_thumbnail(src: Path, t: float, dest: Path, width: int = 368) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.jpg")
    _run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(src),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(tmp),
    ], timeout=60)
    if not tmp.exists():
        raise MediaError("no frame at that time")
    tmp.replace(dest)


def extract_sprite(src: Path, duration: float, dest: Path, columns: int = 10, rows: int = 1) -> None:
    """A columns x rows contact sheet of evenly spaced frames."""
    count = columns * rows
    fps = count / max(duration, 0.1)
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"fps={fps:.5f},scale=160:-2,tile={columns}x{rows}", "-frames:v", "1", "-q:v", "5", str(dest),
    ], timeout=120)


def segment_times(start: float, end: float, n: int) -> np.ndarray:
    """Centre of each of n equal segments in [start, end] (TSN-style segment sampling)."""
    length = max(end - start, 1e-3)
    return (start + (np.arange(n) + 0.5) * length / n).astype(np.float32)


def sample_frames(src: Path, times: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """(T, size, size, 3) uint8 RGB: short side resized to `size`, then centre-cropped."""
    vf = f"scale='if(gt(iw,ih),-2,{size})':'if(gt(iw,ih),{size},-2)',crop={size}:{size}"
    frames = []
    for t in times:
        raw = _run([
            "ffmpeg", "-v", "error", "-ss", f"{float(t):.3f}", "-i", str(src), "-frames:v", "1",
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], timeout=60)
        expected = size * size * 3
        if len(raw) < expected:
            frames.append(frames[-1] if frames else np.zeros((size, size, 3), np.uint8))
        else:
            frames.append(np.frombuffer(raw[:expected], np.uint8).reshape(size, size, 3))
    return np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)


def sample_window_frames(src: Path, start: float, end: float, n: int, size: int = FRAME_SIZE) -> np.ndarray:
    """n frames from one window in a single ffmpeg pass (an fps filter at n / window length)."""
    length = max(end - start, 0.1)
    vf = (
        f"fps={n / length:.6f}:start_time=0,"
        f"scale='if(gt(iw,ih),-2,{size})':'if(gt(iw,ih),{size},-2)',crop={size}:{size}"
    )
    raw = _run([
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], timeout=300)
    frame_bytes = size * size * 3
    count = len(raw) // frame_bytes
    if count == 0:
        return sample_frames(src, segment_times(start, end, n), size)
    frames = np.frombuffer(raw[: count * frame_bytes], np.uint8).reshape(count, size, size, 3)
    idx = np.clip(np.round(np.linspace(0, count - 1, n)).astype(int), 0, count - 1)
    return frames[idx]


def compute_windows(duration: float, window_seconds: float, stride: float, whole_clip: bool) -> list[tuple[float, float]]:
    if whole_clip or duration <= window_seconds:
        return [(0.0, round(duration, 3))]
    windows = []
    n = max(1, math.ceil((duration - window_seconds) / stride) + 1)
    for i in range(n):
        start = min(i * stride, max(duration - window_seconds, 0.0))
        end = min(start + window_seconds, duration)
        if windows and abs(windows[-1][0] - start) < 1e-6:
            continue
        windows.append((round(start, 3), round(end, 3)))
    return windows


def make_test_video(dest: Path, label: str, duration: float, color: str, pattern: str = "testsrc2") -> None:
    """A synthetic clip with a moving test pattern, a colour wash and a text label."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    safe = label.replace("'", "").replace(":", "")
    drawtext = (
        f"drawtext=text='{safe}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.55:"
        f"boxborderw=12:x=(w-text_w)/2:y=h-text_h-28"
    )
    timer = "drawtext=text='%{pts\\:hms}':fontcolor=white:fontsize=18:x=16:y=16:box=1:boxcolor=black@0.4:boxborderw=6"
    blend = "[0][1]blend=all_mode=overlay:all_opacity=0.55"

    def render(filters: str) -> None:
        _run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"{pattern}=size=640x360:rate=24:duration={duration}",
            "-f", "lavfi", "-i", f"color=c={color}:size=640x360:rate=24:duration={duration}",
            "-filter_complex", filters,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dest),
        ], timeout=120)

    try:
        render(f"{blend},{drawtext},{timer}")
    except MediaError:
        render(blend)  # ffmpeg builds without drawtext/fonts still get a usable clip
