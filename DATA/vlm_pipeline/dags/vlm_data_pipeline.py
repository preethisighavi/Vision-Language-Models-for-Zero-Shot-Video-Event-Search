"""
vlm_data_pipeline.py
────────────────────
End-to-end Airflow DAG for VLM (CLIP / BLIP) data preparation.

Datasets: MSR-VTT + MSVD
Tasks:
  1. ingest          – Load JSON, unify schema, output raw parquet
  2. text_normalize  – Clean captions, remove short/noisy ones
  3. dedup_filter    – Drop exact duplicate (video_id, caption) pairs
  4. quality_gate    – CLIP similarity filter; remove outlier pairs
  5. keyframe_extract– Extract 3 frames per video as JPEG
  6. build_splits    – Construct train / val / test splits (video-level)
  7. format_clip     – Write WebDataset .tar shards for CLIP fine-tuning
  8. format_blip     – Write COCO-style annotation JSON for BLIP
  9. validate        – Integrity check + summary report
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── task imports (tasks/ is on PYTHONPATH via docker-compose env) ─────────────
sys.path.insert(0, "/opt/airflow/tasks")

import ingest as _ingest
import text_normalize as _text
import quality_filter as _quality
import keyframe_extract as _frames
import build_splits as _splits
import validate as _validate
import export_results as _export

# ─────────────────────────────────────────────────────────────────────────────
# Default arguments
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner": "vlm-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

CFG_PATH = "/opt/airflow/config/pipeline_config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="vlm_data_pipeline",
    description="End-to-end data pipeline: MSR-VTT + MSVD → CLIP & BLIP formats",
    default_args=default_args,
    schedule_interval=None,          # Triggered manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vlm", "clip", "blip", "data-pipeline"],
    max_active_runs=1,
) as dag:

    # ── 1. Ingest ─────────────────────────────────────────────────────────────
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        **Ingest** raw JSON files from both MSR-VTT and MSVD.
        Normalizes schema differences and writes `raw_unified.parquet`.
        """,
    )

    # ── 2. Text normalization ─────────────────────────────────────────────────
    text_norm = PythonOperator(
        task_id="text_normalize",
        python_callable=_text.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Lowercase, strip noise, filter short captions (<3 words).
        Writes `text_normalized.parquet`.
        """,
    )

    # ── 3. Deduplication ─────────────────────────────────────────────────────
    dedup = PythonOperator(
        task_id="dedup_filter",
        python_callable=_text.run_dedup,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Removes exact duplicate (video_id, caption) pairs globally and
        intra-video. Writes `deduped.parquet`.
        """,
    )

    # ── 4. CLIP quality gate ──────────────────────────────────────────────────
    quality = PythonOperator(
        task_id="quality_gate",
        python_callable=_quality.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Runs CLIP ViT-B/32 similarity on each (frame, caption) pair.
        Drops video_ids where all captions score < 0.15. 
        Writes `quality_filtered.parquet` + similarity scores.
        """,
    )

    # ── 5. Keyframe extraction ────────────────────────────────────────────────
    frames = PythonOperator(
        task_id="keyframe_extract",
        python_callable=_frames.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Extracts 3 equidistant frames (begin/mid/end) per video.
        Saves 224×224 JPEGs to `intermediate/keyframes/`.
        Updates DataFrame with `frame_paths` column.
        """,
    )

    # ── 6. Split construction ─────────────────────────────────────────────────
    splits = PythonOperator(
        task_id="build_splits",
        python_callable=_splits.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Constructs video-level train/val/test splits.
        - MSR-VTT: stratified by category (90/10 train/val + official test 1K)
        - MSVD: honors official split files
        Writes `split_train.parquet`, `split_val.parquet`, `split_test.parquet`.
        """,
    )

    # ── 9. Validation ─────────────────────────────────────────────────────────
    validate = PythonOperator(
        task_id="validate_outputs",
        python_callable=_validate.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Validates integrity of all outputs:
        - No video_id overlap across splits
        - Frame files exist for all records
        - BLIP JSON structurally valid
        - Summary report → `outputs/pipeline_report.json`
        """,
    )

    # ── 10. Export to Drive ───────────────────────────────────────────────────
    export = PythonOperator(
        task_id="export_results",
        python_callable=_export.run,
        op_kwargs={"config_path": CFG_PATH},
        doc_md="""
        Sync final `outputs/` folder to Google Drive for persistence.
        Uses rsync to ensure fast, safe transfer to FUSE.
        """,
    )

    # ─── Task dependency chain ────────────────────────────────────────────────
    (
        ingest
        >> text_norm
        >> dedup
        >> quality
        >> frames
        >> splits
        >> validate
        >> export
    )
