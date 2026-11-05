"""Clip manifest for MSR-VTT + MSVD: one row per video with its captions and file path.

Reuses the annotation loaders from the team's data pipeline
(DATA/vlm_pipeline/tasks/ingest.py) but not its batch/demo sampling, so every
available video is included. Splits:
  MSR-VTT: official 1k test set -> test, everything else -> train (5% held out as val)
  MSVD:    official train/val/test files
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov")


def _index_videos(dirs: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                found.setdefault(p.stem, p)
    return found


def build_manifest(msrvtt_dir: str, msvd_dir: str, val_ratio: float = 0.05, seed: int = 42) -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "DATA" / "vlm_pipeline"))
    from tasks.ingest import _load_msrvtt, _load_msvd  # noqa: E402

    df = pd.concat([_load_msrvtt(msrvtt_dir), _load_msvd(msvd_dir)], ignore_index=True)
    df = df[df["caption"].str.strip().str.len() > 0]

    rows = (
        df.groupby(["video_id", "source"])
        .agg(captions=("caption", lambda s: sorted(set(map(str.strip, s)))), split_hint=("split_hint", "first"))
        .reset_index()
    )
    # a video listed in the MSR-VTT test file must stay in test even if it also appears in train json
    test_ids = set(df.loc[(df["source"] == "MSR-VTT") & (df["split_hint"] == "test"), "video_id"])
    rows.loc[rows["video_id"].isin(test_ids) & (rows["source"] == "MSR-VTT"), "split_hint"] = "test"

    rows["split"] = rows["split_hint"]
    vtt_train = rows[(rows["source"] == "MSR-VTT") & (rows["split"] == "train")]
    val_ids = vtt_train.sample(frac=val_ratio, random_state=seed)["video_id"]
    rows.loc[(rows["source"] == "MSR-VTT") & rows["video_id"].isin(val_ids), "split"] = "val"

    videos = _index_videos([Path(msrvtt_dir), Path(msvd_dir)])
    rows["path"] = rows["video_id"].map(lambda v: str(videos[v]) if v in videos else None)
    missing = rows["path"].isna().sum()
    if missing:
        print(f"[manifest] {missing} of {len(rows)} videos have annotations but no video file; dropped")
    rows = rows.dropna(subset=["path"]).drop(columns=["split_hint"])
    return rows.sort_values(["source", "video_id"]).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write the T2V clip manifest")
    ap.add_argument("--msrvtt-dir", required=True)
    ap.add_argument("--msvd-dir", required=True)
    ap.add_argument("--out", default="T2V/data/manifest.parquet")
    a = ap.parse_args()
    m = build_manifest(a.msrvtt_dir, a.msvd_dir)
    m.to_parquet(a.out, index=False)
    print(m.groupby(["source", "split"]).size())
