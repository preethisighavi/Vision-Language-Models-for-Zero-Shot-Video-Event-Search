"""
quality_filter.py
─────────────────
Task 4: CLIP-based quality gate.
Uses PyAV (av) for video decoding — ARM64/cross-platform compatible.

For each (video_id, caption) pair:
  1. Extract the middle frame from the raw video file.
  2. Compute cosine similarity via openai/clip-vit-base-patch32.
  3. Store similarity score on each row.
  4. Drop any video_id where the MAXIMUM similarity across all its
     captions is below the configured threshold (default 0.15).

EDA finding: ~N% of pairs had max-sim < 0.15, indicating misaligned
or mislabeled samples that would harm VLM training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

log = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_video_path(video_id: str, source: str, msrvtt_dir: str, msvd_dir: str) -> str:
    """Return local path to video."""
    dirs = [msrvtt_dir if source == "MSR-VTT" else msvd_dir]
    for d in dirs:
        for ext in [".mp4", ".avi", ".mkv"]:
            p = os.path.join(d, "raw_videos", f"{video_id}{ext}")
            if os.path.exists(p):
                return p
    return ""


def _extract_middle_frame(video_path: str) -> Image.Image | None:
    """Extract the middle frame of a video using PyAV."""
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        target = max(0, total_frames // 2)
        container.seek(target, stream=stream)
        for frame in container.decode(stream):
            img = frame.to_image()  # PIL Image
            container.close()
            return img
        container.close()
        return None
    except Exception as e:
        log.warning("Failed to extract frame from %s: %s", video_path, str(e))
        return None


def _compute_similarities_batch(
    records: list[dict],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_size: int,
    msrvtt_dir: str,
    msvd_dir: str,
) -> list[float]:
    """Compute CLIP cosine similarities for a batch of (frame_image, caption) pairs."""
    similarities = []

    for i in tqdm(range(0, len(records), batch_size), desc="CLIP audit"):
        batch = records[i : i + batch_size]
        texts = []
        images = []
        valid_indices = []

        for j, rec in enumerate(batch):
            vpath = _get_video_path(rec["video_id"], rec["source"], msrvtt_dir, msvd_dir)
            
            if not vpath:
                log.debug("No video path found for %s (%s)", rec["video_id"], rec["source"])
                continue
            
            # Extract frame
            img = _extract_middle_frame(vpath)
            if img is None:
                log.debug("Frame extraction returned None for %s", vpath)
                continue
            texts.append(rec["caption"])
            images.append(img)
            valid_indices.append(j)

        batch_results: dict[int, float] = {}

        if texts:
            try:
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    sims = (img_emb * txt_emb).sum(dim=-1).cpu().numpy()

                for vi, sim in zip(valid_indices, sims):
                    batch_results[vi] = float(sim)

            except Exception as e:
                log.warning("Batch failed: %s", e)

        # Append exactly one result (float or None) per record in batch
        for j in range(len(batch)):
            similarities.append(batch_results.get(j, None))

    return similarities


def run(config_path: str, **kwargs) -> str:
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    msrvtt_dir = cfg["paths"]["msrvtt_dir"]
    msvd_dir = cfg["paths"]["msvd_dir"]
    q_cfg = cfg["quality"]

    clip_model_name = q_cfg.get("clip_model", "openai/clip-vit-base-patch32")
    threshold = q_cfg.get("outlier_threshold", 0.15)
    batch_size = q_cfg.get("batch_size", 64)
    sample_limit = q_cfg.get("sample_for_audit", None)

    in_path = os.path.join(intermediate_dir, "deduped.parquet")
    df = pd.read_parquet(in_path)

    if sample_limit:
        # Smoke-test mode: limit to N videos
        sampled_vids = df["video_id"].unique()[:sample_limit]
        df = df[df["video_id"].isin(sampled_vids)].reset_index(drop=True)
        log.info("Smoke-test mode: limiting to %d videos", len(sampled_vids))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading CLIP model %s on %s", clip_model_name, device)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    records = df.to_dict("records")
    similarities = _compute_similarities_batch(
        records, model, processor, device, batch_size, msrvtt_dir, msvd_dir
    )

    df["clip_sim"] = similarities

    # Drop rows where similarity couldn't be computed (missing video file)
    before_missing = len(df)
    df = df[df["clip_sim"].notna()].reset_index(drop=True)
    log.info(
        "Dropped %d rows with missing video files", before_missing - len(df)
    )

    if df.empty:
        log.warning("No rows retained after missing-file filtering. Returning empty results.")
        out_path = os.path.join(intermediate_dir, "quality_filtered.parquet")
        df.to_parquet(out_path, index=False)
        return out_path

    # Compute max similarity per video_id and flag outliers
    max_sim = df.groupby("video_id")["clip_sim"].max()
    outlier_vids = max_sim[max_sim < threshold].index
    log.info(
        "Outlier videos (max_sim < %.2f): %d / %d",
        threshold, len(outlier_vids), df["video_id"].nunique(),
    )

    df = df[~df["video_id"].isin(outlier_vids)].reset_index(drop=True)
    log.info("Retained %d rows after quality filtering", len(df))

    out_path = os.path.join(intermediate_dir, "quality_filtered.parquet")
    df.to_parquet(out_path, index=False)
    log.info("Written: %s", out_path)
    return out_path
