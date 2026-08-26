"""
build_splits.py
───────────────
Task 6: Construct train / val / test splits at the VIDEO level.

Strategy:
  MSR-VTT:
    - Test  = official msrvtt_test_1k.json video IDs (split_hint='test')
    - Train = 90% of remaining (stratified by category)
    - Val   = 10% of remaining (stratified by category)

  MSVD:
    - Train / Val / Test = honor official split_hint from ingest step
      (msvd_train.json / msvd_val.json / msvd_test.json)

Output:
  split_train.parquet, split_val.parquet, split_test.parquet
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _split_msrvtt(df_vtt: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split MSR-VTT into train/val/test at the video_id level.
    Test set = videos already tagged split_hint='test'.
    Train/val = stratified split of the rest by category.
    """
    df_test = df_vtt[df_vtt["split_hint"] == "test"]
    df_rest = df_vtt[df_vtt["split_hint"] != "test"]

    # Video-level split (to prevent leakage)
    vid_meta = (
        df_rest[["video_id", "category"]]
        .drop_duplicates("video_id")
        .reset_index(drop=True)
    )

    if vid_meta.empty:
        log.warning("No training/validation samples found for MSR-VTT. Returning empty splits.")
        # Return empty DFs with correct columns for train/val, and move everything to test
        df_train = pd.DataFrame(columns=df_vtt.columns.tolist() + ["split"])
        df_val = pd.DataFrame(columns=df_vtt.columns.tolist() + ["split"])
        df_test = df_test.copy()
        df_test["split"] = "test"
        return df_train, df_val, df_test

    # Stratify by category; fall back to random if a category has <2 members
    try:
        train_vids, val_vids = train_test_split(
            vid_meta["video_id"],
            test_size=val_ratio,
            stratify=vid_meta["category"],
            random_state=seed,
        )
    except ValueError:
        log.warning("Stratified split failed (few samples), falling back to random split")
        train_vids, val_vids = train_test_split(
            vid_meta["video_id"],
            test_size=val_ratio,
            random_state=seed,
        )

    df_train = df_rest[df_rest["video_id"].isin(train_vids)].copy()
    df_val = df_rest[df_rest["video_id"].isin(val_vids)].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test = df_test.copy()
    df_test["split"] = "test"

    return df_train, df_val, df_test


def _split_msvd(df_msvd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Honor the official MSVD split_hint from ingest."""
    df_train = df_msvd[df_msvd["split_hint"] == "train"].copy()
    df_val = df_msvd[df_msvd["split_hint"] == "val"].copy()
    df_test = df_msvd[df_msvd["split_hint"] == "test"].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    return df_train, df_val, df_test


def run(config_path: str, **kwargs) -> dict[str, str]:
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    split_cfg = cfg["splits"]
    val_ratio = split_cfg.get("msrvtt_val_ratio", 0.10)
    seed = split_cfg.get("random_seed", 42)

    # Prefer with_frames.parquet; fall back to deduped if frames are missing
    frames_path = os.path.join(intermediate_dir, "with_frames.parquet")
    fallback_path = os.path.join(intermediate_dir, "deduped.parquet")

    if os.path.exists(frames_path):
        df = pd.read_parquet(frames_path)
    else:
        df = pd.read_parquet(fallback_path)
        df["frame_paths"] = [[] for _ in range(len(df))]

    # Guard: if empty or missing expected columns, fall back gracefully
    required_cols = {"video_id", "caption", "source", "split_hint"}
    if df.empty or not required_cols.issubset(df.columns):
        log.warning(
            "with_frames.parquet is empty or missing columns %s — "
            "falling back to deduped.parquet",
            required_cols - set(df.columns),
        )
        df = pd.read_parquet(fallback_path)
        df["frame_paths"] = [[] for _ in range(len(df))]

    log.info(
        "Building splits on %d rows | %d unique videos | columns: %s",
        len(df), df["video_id"].nunique(), list(df.columns),
    )

    # Split by source
    df_vtt = df[df["source"] == "MSR-VTT"]
    df_msvd = df[df["source"] == "MSVD"]

    log.info("Splitting MSR-VTT (%d rows)...", len(df_vtt))
    vtt_train, vtt_val, vtt_test = _split_msrvtt(df_vtt, val_ratio, seed)

    log.info("Splitting MSVD (%d rows)...", len(df_msvd))
    msvd_train, msvd_val, msvd_test = _split_msvd(df_msvd)

    # Combine
    df_train = pd.concat([vtt_train, msvd_train], ignore_index=True)
    df_val = pd.concat([vtt_val, msvd_val], ignore_index=True)
    df_test = pd.concat([vtt_test, msvd_test], ignore_index=True)

    # Log summary
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        vids = split_df["video_id"].nunique()
        rows = len(split_df)
        srcs = split_df["source"].value_counts().to_dict()
        log.info("  %s: %d rows | %d unique videos | %s", name, rows, vids, srcs)

    # Assert no video_id overlap
    all_splits = [df_train, df_val, df_test]
    all_vids = [set(s["video_id"].unique()) for s in all_splits]
    assert all_vids[0].isdisjoint(all_vids[1]), "Train/Val video_id LEAK!"
    assert all_vids[0].isdisjoint(all_vids[2]), "Train/Test video_id LEAK!"
    assert all_vids[1].isdisjoint(all_vids[2]), "Val/Test video_id LEAK!"
    log.info("No video_id leakage confirmed across splits.")

    # Save
    paths = {}
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out = os.path.join(intermediate_dir, f"split_{name}.parquet")
        split_df.to_parquet(out, index=False)
        paths[name] = out
        log.info("Written: %s", out)

    return paths
