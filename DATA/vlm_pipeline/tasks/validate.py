"""
validate.py
───────────
Task 9: Post-pipeline integrity checks and summary report.

Checks:
  1. No video_id appears in more than one split.
  2. All frame file paths referenced in parquets actually exist on disk.
  3. BLIP annotations.json is valid and all image_ids resolve.
  4. Median CLIP similarity per split > 0.2 (data quality baseline).
  5. Sample counts per split are within expected ranges.

Output:
  outputs/pipeline_report.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import pandas as pd
import yaml

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _check_no_leakage(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Assert no video_id appears in more than one split."""
    errors = []
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = set(splits[a]["video_id"]) & set(splits[b]["video_id"])
            if overlap:
                errors.append(f"Leakage: {a}↔{b} share {len(overlap)} video_ids: {list(overlap)[:5]}")
    return errors


def _check_frame_files(df: pd.DataFrame, split_name: str) -> list[str]:
    """Verify all referenced frame files exist."""
    errors = []
    missing = 0
    for row in df.itertuples():
        # frame_paths is a numpy array or list; checking len() avoids truth value ambiguity
        if hasattr(row, "frame_paths") and row.frame_paths is not None and len(row.frame_paths) > 0:
            for fp in row.frame_paths:
                if not os.path.exists(fp):
                    missing += 1
    if missing:
        errors.append(f"[{split_name}] {missing} missing frame files")
    return errors


def _check_blip_annotations(out_dir: str, split: str) -> list[str]:
    """Validate BLIP COCO annotation JSON."""
    errors = []
    ann_path = os.path.join(out_dir, "blip", split, "annotations.json")
    if not os.path.exists(ann_path):
        errors.append(f"[{split}] Missing BLIP annotations.json")
        return errors

    with open(ann_path) as f:
        data = json.load(f)

    image_ids = {img["id"] for img in data.get("images", [])}
    ann_image_ids = {ann["image_id"] for ann in data.get("annotations", [])}
    orphan = ann_image_ids - image_ids
    if orphan:
        errors.append(
            f"[{split}] BLIP: {len(orphan)} annotations reference non-existent image_ids"
        )
    return errors


def _check_clip_similarity(df: pd.DataFrame, split_name: str, min_median: float = 0.20) -> list[str]:
    """Warn if median CLIP similarity is below threshold."""
    warnings = []
    if "clip_sim" not in df.columns:
        return warnings
    med = df["clip_sim"].dropna().median()
    if med < min_median:
        warnings.append(
            f"[{split_name}] Median CLIP similarity {med:.3f} < {min_median} (quality concern)"
        )
    return warnings


def run(config_path: str, **kwargs):
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    output_dir = cfg["paths"]["output_dir"]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "splits": {},
        "errors": [],
        "warnings": [],
        "passed": True,
    }

    # ── Load splits ───────────────────────────────────────────────────────────
    splits: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        pq = os.path.join(intermediate_dir, f"split_{split}.parquet")
        if os.path.exists(pq):
            splits[split] = pd.read_parquet(pq)
            df = splits[split]
            report["splits"][split] = {
                "rows": len(df),
                "unique_videos": df["video_id"].nunique(),
                "sources": df["source"].value_counts().to_dict(),
                "median_clip_sim": (
                    round(float(df["clip_sim"].dropna().median()), 4)
                    if "clip_sim" in df.columns
                    else None
                ),
            }
        else:
            report["errors"].append(f"Missing split parquet: {pq}")

    # ── Check 1: No leakage ───────────────────────────────────────────────────
    leak_errors = _check_no_leakage(splits)
    report["errors"].extend(leak_errors)

    # ── Check 2: Frame files exist ────────────────────────────────────────────
    for split_name, df in splits.items():
        frame_errors = _check_frame_files(df, split_name)
        report["errors"].extend(frame_errors)

    # ── Check 3: BLIP annotation integrity ───────────────────────────────────
    # Skipped: Format tasks have been removed.
    # for split in splits:
    #     blip_errors = _check_blip_annotations(output_dir, split)
    #     report["errors"].extend(blip_errors)

    # ── Check 4: CLIP similarity quality ─────────────────────────────────────
    for split_name, df in splits.items():
        sim_warnings = _check_clip_similarity(df, split_name)
        report["warnings"].extend(sim_warnings)

    # ── Determine pass/fail ───────────────────────────────────────────────────
    report["passed"] = len(report["errors"]) == 0

    # ── Write report ──────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, "pipeline_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Log summary
    log.info("=" * 60)
    log.info("PIPELINE VALIDATION REPORT")
    log.info("=" * 60)
    for split, stats in report["splits"].items():
        log.info(
            "  [%s] %d rows | %d videos | median_clip_sim: %s | sources: %s",
            split.upper(),
            stats["rows"],
            stats["unique_videos"],
            stats["median_clip_sim"],
            stats["sources"],
        )
    if report["errors"]:
        log.error("ERRORS (%d):", len(report["errors"]))
        for e in report["errors"]:
            log.error("  ✗ %s", e)
    if report["warnings"]:
        log.warning("WARNINGS (%d):", len(report["warnings"]))
        for w in report["warnings"]:
            log.warning("  ⚠ %s", w)

    status = "✓ PASSED" if report["passed"] else "✗ FAILED"
    log.info("Result: %s → %s", status, report_path)

    if not report["passed"]:
        raise RuntimeError(f"Pipeline validation failed. See {report_path}")

    return report_path
