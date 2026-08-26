"""
ingest.py
─────────
Task 1: Load MSR-VTT and MSVD JSON annotation files from the local
data_local/ directory, normalize schema, and write a unified flat Parquet.

Data resolution order for each JSON file:
  1. data_local/MSRVTT/ or data_local/MSVD/  (Docker-mounted local copy)
  2. Fallback to the original Google Drive FUSE path (if accessible)

Video files live under raw_videos/ inside the same directory and are
used downstream by quality_filter and keyframe_extract tasks — ingest
itself only reads JSON annotations.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# ── Candidate base directories tried in order ─────────────────────────────────
# 1. data_local/  (local copy, reliable, no FUSE issues)
# 2. original path set by env var (Google Drive FUSE mount — may hit OSError 35)
_LOCAL_BASE = "/data"   # Mounted from ./data_local via docker-compose


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _safe_read_json(path: str) -> list | dict | None:
    """Read a JSON file, returning None on any IO error."""
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as e:
        log.warning("Could not read %s: %s", path, e)
        return None


def _resolve_path(configured_dir: str, filename: str) -> str | None:
    """
    Try to find 'filename' by checking:
      1. configured_dir as-is        (data_local/ mount inside Docker)
      2. configured_dir parent swap  (handles any path prefix differences)
    Returns the first path that exists, else None.
    """
    candidate = os.path.join(configured_dir, filename)
    if os.path.exists(candidate):
        return candidate

    # Try swapping the base prefix to _LOCAL_BASE
    folder_name = os.path.basename(configured_dir)  # e.g. "MSRVTT"
    alt = os.path.join(_LOCAL_BASE, folder_name, filename)
    if os.path.exists(alt):
        return alt

    log.warning("File not found in any location: %s", filename)
    return None


def _load_msrvtt(msrvtt_dir: str) -> pd.DataFrame:
    rows = []

    # Discover all annotation JSON files
    json_files = sorted(glob.glob(os.path.join(msrvtt_dir, "msrvtt_*.json")))
    if not json_files:
        # Try the _LOCAL_BASE fallback location
        alt_dir = os.path.join(_LOCAL_BASE, "MSRVTT")
        json_files = sorted(glob.glob(os.path.join(alt_dir, "msrvtt_*.json")))

    log.info("MSR-VTT: found %d JSON file(s) in %s", len(json_files), msrvtt_dir)

    for fpath in json_files:
        fname = os.path.basename(fpath)
        split_hint = "test" if "test" in fname else "train"

        content = _safe_read_json(fpath)
        if content is None:
            continue

        # Build video_id → category map (only present in full train JSON)
        cat_map: dict[str, str] = {}
        if isinstance(content, dict) and "videos" in content:
            for v in content["videos"]:
                cat_map[v["video_id"]] = v.get("category", "unknown")

        # Extract sentence rows
        sentences = []
        if isinstance(content, dict) and "sentences" in content:
            sentences = content["sentences"]
        elif isinstance(content, list):
            sentences = content   # test_1k.json is a plain list

        for s in sentences:
            vid = str(s.get("video_id") or s.get("clip_name", ""))
            rows.append({
                "video_id": vid,
                "caption": str(s.get("caption", "")),
                "source": "MSR-VTT",
                "category": cat_map.get(vid, "unknown"),
                "split_hint": split_hint,
            })

    df = pd.DataFrame(rows)
    log.info(
        "MSR-VTT: %d captions | %d unique videos",
        len(df), df["video_id"].nunique() if not df.empty else 0,
    )
    return df


def _load_msvd(msvd_dir: str) -> pd.DataFrame:
    rows = []
    split_files = {
        "train": "msvd_train.json",
        "val":   "msvd_val.json",
        "test":  "msvd_test.json",
    }

    for split, fname in split_files.items():
        fpath = os.path.join(msvd_dir, fname)
        if not os.path.exists(fpath):
            # Try local base fallback
            fpath = os.path.join(_LOCAL_BASE, "MSVD", fname)

        content = _safe_read_json(fpath)
        if content is None:
            continue

        for item in content:
            vid = str(item.get("video_id", ""))
            caps = item.get("caption", [])
            if isinstance(caps, str):
                caps = [caps]
            for cap in caps:
                rows.append({
                    "video_id": vid,
                    "caption": str(cap),
                    "source": "MSVD",
                    "category": "N/A",
                    "split_hint": split,
                })

    df = pd.DataFrame(rows)
    log.info(
        "MSVD: %d captions | %d unique videos",
        len(df), df["video_id"].nunique() if not df.empty else 0,
    )
    return df


def run(config_path: str, **kwargs) -> str:
    cfg = _load_config(config_path)
    msrvtt_dir = cfg["paths"]["msrvtt_dir"]
    msvd_dir   = cfg["paths"]["msvd_dir"]
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    os.makedirs(intermediate_dir, exist_ok=True)

    df_msrvtt = _load_msrvtt(msrvtt_dir)
    df_msvd   = _load_msvd(msvd_dir)

    df_all = pd.concat([df_msrvtt, df_msvd], ignore_index=True)

    # ── Orchestrator Batch Filter ──
    # Limit the massive metadata dataframe to ONLY the videos currently staged in the local drive.
    valid_vid_ids = set()
    for ds_dir in [msrvtt_dir, msvd_dir]:
        raw_dir = os.path.join(ds_dir, "raw_videos")
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                if f.endswith((".mp4", ".avi", ".mkv")):
                    valid_vid_ids.add(f.rsplit(".", 1)[0])

    initial_len = len(df_all)
    df_all = df_all[df_all["video_id"].isin(valid_vid_ids)]

    # ── HOTFIX FOR DEMO: Balanced 2-video Split Sampling ──
    limit = 2
    sampled_dfs = []
    for hint in df_all["split_hint"].unique():
        hint_df = df_all[df_all["split_hint"] == hint]
        # Multiplier to train so train_test_split has enough videos
        multiplier = 2 if hint == "train" else 1
        vids = hint_df["video_id"].unique()[:limit * multiplier]
        sampled_dfs.append(hint_df[hint_df["video_id"].isin(vids)])
        
    df_all = pd.concat(sampled_dfs, ignore_index=True)
    log.info(
        "Orchestrator filter: Staged batch handles %d records out of full %d.",
        len(df_all), initial_len
    )

    if df_all.empty:
        raise RuntimeError(
            "No data loaded! Ensure data_local/MSRVTT/ and data_local/MSVD/ "
            "contain the JSON annotation files."
        )

    log.info(
        "Unified dataset: %d total captions | %d unique videos | sources: %s",
        len(df_all),
        df_all["video_id"].nunique(),
        df_all["source"].value_counts().to_dict(),
    )

    out_path = os.path.join(intermediate_dir, "raw_unified.parquet")
    df_all.to_parquet(out_path, index=False)
    log.info("Written → %s", out_path)
    return out_path
