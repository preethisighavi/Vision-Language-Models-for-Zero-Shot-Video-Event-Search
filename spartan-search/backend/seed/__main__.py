"""python -m seed [--force] [--collection ID]

Registers the default collections and indexes every video found in
DATA_DIR/seed/<collection>/ with every enabled model. If no seed videos exist,
30 synthetic clips are generated first.
"""

import argparse
import sys

from app.config import get_settings
from app.db.session import init_db, session_scope
from app.services import media
from app.services.catalog import ensure_default_collections
from app.services.library_service import VIDEO_SUFFIXES, import_folder
from seed import synthetic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-index videos that are already ready")
    parser.add_argument("--collection", help="only seed this collection")
    args = parser.parse_args()

    if not media.ffmpeg_available():
        print("ffmpeg/ffprobe not found. Install ffmpeg or run inside the Docker image.", file=sys.stderr)
        return 1

    settings = get_settings()
    for path in (settings.data_dir, settings.media_dir, settings.index_dir, settings.features_dir, settings.seed_dir):
        path.mkdir(parents=True, exist_ok=True)
    init_db()
    with session_scope() as s:
        ensure_default_collections(s)

    has_videos = any(p.suffix.lower() in VIDEO_SUFFIXES for p in settings.seed_dir.rglob("*"))
    if not has_videos:
        print(f"No videos in {settings.seed_dir}; generating synthetic clips.")
        synthetic.generate(settings.seed_dir)

    folders = sorted(p for p in settings.seed_dir.iterdir() if p.is_dir())
    if args.collection:
        folders = [p for p in folders if p.name == args.collection]
    total = 0
    for folder in folders:
        print(f"Indexing {folder.name} ...")

        def progress(i, n, video_id, skipped):
            print(f"  [{i}/{n}] {video_id}{' (already indexed)' if skipped else ''}")

        total += import_folder(folder.name, folder, force=args.force, on_progress=progress)
    print(f"Done. Indexed {total} video(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
