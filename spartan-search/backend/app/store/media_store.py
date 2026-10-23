"""Video files, thumbnails and posters on local disk.

The interface (put_video / open_range / put_thumbnail) is what an S3 or GCS
implementation would need to provide later.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

from app.config import get_settings

CHUNK = 1024 * 256


class MediaStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or get_settings().media_dir)
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        (self.root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.root / "thumbs").mkdir(parents=True, exist_ok=True)

    # paths
    def video_path(self, video_id: str) -> Path:
        return self.root / "videos" / f"{video_id}.mp4"

    def upload_path(self, video_id: str, suffix: str) -> Path:
        return self.root / "uploads" / f"{video_id}{suffix}"

    def thumb_dir(self, video_id: str) -> Path:
        return self.root / "thumbs" / video_id

    def thumbnail_path(self, video_id: str, t: float) -> Path:
        return self.thumb_dir(video_id) / f"t{int(round(t * 10)):06d}.jpg"

    def poster_path(self, video_id: str) -> Path:
        return self.thumb_dir(video_id) / "poster.jpg"

    def sprite_path(self, video_id: str) -> Path:
        return self.thumb_dir(video_id) / "sprite.jpg"

    # operations
    def put_video(self, video_id: str, source: Path) -> Path:
        dest = self.video_path(video_id)
        if Path(source) != dest:
            shutil.copyfile(source, dest)
        return dest

    def has_video(self, video_id: str) -> bool:
        return self.video_path(video_id).exists()

    def size(self, video_id: str) -> int:
        return self.video_path(video_id).stat().st_size

    def open_range(self, video_id: str, start: int, end: int) -> Iterator[bytes]:
        """Yields bytes [start, end] inclusive."""
        remaining = end - start + 1
        with open(self.video_path(video_id), "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    def put_thumbnail(self, video_id: str, t: float, data: bytes) -> Path:
        path = self.thumbnail_path(video_id, t)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def delete_video(self, video_id: str) -> None:
        self.video_path(video_id).unlink(missing_ok=True)
        for p in (self.root / "uploads").glob(f"{video_id}.*"):
            p.unlink(missing_ok=True)
        shutil.rmtree(self.thumb_dir(video_id), ignore_errors=True)
