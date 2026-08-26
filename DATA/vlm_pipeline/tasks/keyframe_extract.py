"""
keyframe_extract.py
───────────────────
Task 5: Extract frames from raw video files.

Per video, extracts 3 equidistant frames (beginning / middle / end)
and saves them as 224×224 JPEGs.

Column added to DataFrame: `frame_paths` → list of 3 absolute paths
(or fewer if video is too short).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_video_path(video_id: str, source: str, msrvtt_dir: str, msvd_dir: str) -> str:
    base_dir = msrvtt_dir if source == "MSR-VTT" else msvd_dir
    for ext in [".mp4", ".avi", ".mkv"]:
        p = os.path.join(base_dir, "raw_videos", f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return ""


def _extract_frames(
    video_path: str, n_frames: int, output_dir: str, video_id: str, size: tuple[int, int]
) -> list[str]:
    """Extract n equidistant frames from video using PyAV, save as JPEG, return paths."""
    try:
        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames

        # If frame count not in metadata, decode to count
        if not total or total == 0:
            total = sum(1 for _ in container.decode(stream))
            container.seek(0)

        indices = set(np.linspace(0, max(total - 1, 0), n_frames, dtype=int).tolist())
        saved_paths = []
        seen = 0
        frame_idx = 0

        for frame in container.decode(container.streams.video[0]):
            if frame_idx in indices:
                img = frame.to_image().convert("RGB").resize(size, Image.LANCZOS)
                fname = f"{video_id}_frame_{len(saved_paths)}.jpg"
                out_path = os.path.join(output_dir, fname)
                img.save(out_path, format="JPEG", quality=95)
                saved_paths.append(out_path)
                seen += 1
                if seen >= n_frames:
                    break
            frame_idx += 1

        container.close()
        return saved_paths

    except Exception as e:
        log.debug("Frame extraction failed for %s: %s", video_path, e)
        return []


def run(config_path: str, **kwargs) -> str:
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    keyframe_dir = cfg["paths"]["keyframe_dir"]
    msrvtt_dir = cfg["paths"]["msrvtt_dir"]
    msvd_dir = cfg["paths"]["msvd_dir"]
    kf_cfg = cfg["keyframes"]

    n_frames = kf_cfg.get("frames_per_video", 3)
    size = tuple(kf_cfg.get("output_size", [224, 224]))
    os.makedirs(keyframe_dir, exist_ok=True)

    in_path = os.path.join(intermediate_dir, "quality_filtered.parquet")
    df = pd.read_parquet(in_path)

    # Work at video level (not row level) to avoid redundant extraction
    unique_vids = df[["video_id", "source"]].drop_duplicates()
    log.info("Extracting keyframes for %d unique videos...", len(unique_vids))

    video_to_frames: dict[str, list[str]] = {}
    failed = 0

    for _, row in tqdm(unique_vids.iterrows(), total=len(unique_vids), desc="Keyframes"):
        vid = row["video_id"]
        src = row["source"]
        vpath = _get_video_path(vid, src, msrvtt_dir, msvd_dir)

        if not vpath:
            video_to_frames[vid] = []
            failed += 1
            continue

        frames = _extract_frames(vpath, n_frames, keyframe_dir, vid, size)
        video_to_frames[vid] = frames

    log.info(
        "Keyframe extraction complete. Failed: %d / %d videos",
        failed, len(unique_vids),
    )

    # Attach frame paths to each row
    df["frame_paths"] = df["video_id"].map(video_to_frames)

    # Drop rows for videos with no frames (missing files)
    before = len(df)
    df = df[df["frame_paths"].apply(lambda fp: len(fp) > 0)].reset_index(drop=True)
    log.info("Dropped %d rows with no extracted frames", before - len(df))

    # Save frame mapping as JSON (useful for debugging)
    mapping_path = os.path.join(intermediate_dir, "video_frame_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(video_to_frames, f, indent=2)

    out_path = os.path.join(intermediate_dir, "with_frames.parquet")
    df.to_parquet(out_path, index=False)
    log.info("Written: %s (%d rows)", out_path, len(df))
    return out_path
