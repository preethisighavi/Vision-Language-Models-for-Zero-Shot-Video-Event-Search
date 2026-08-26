"""
text_normalize.py
─────────────────
Task 2: Clean and normalize captions.
Task 3: Deduplicate (video_id, caption) pairs.

Normalization rules (from EDA):
  - Lowercase
  - Strip leading/trailing whitespace
  - Drop captions with < 3 words (noisy short captions observed in MSVD)
  - Drop captions with > 50 words (outliers)
  - Drop all-stopword captions
"""

from __future__ import annotations

import logging
import os
import re
import string

import nltk
import pandas as pd
import yaml

log = logging.getLogger(__name__)

# Download NLTK data (runs once in the container)
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _normalize_caption(text: str, lowercase: bool = True) -> str:
    """Apply basic text normalization."""
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _is_valid(caption: str, min_words: int, max_words: int) -> bool:
    """Return False if the caption is too short, too long, or all stopwords."""
    tokens = caption.split()
    if len(tokens) < min_words or len(tokens) > max_words:
        return False
    # Reject if every non-punctuation token is a stopword
    content_tokens = [t.strip(string.punctuation) for t in tokens]
    content_tokens = [t for t in content_tokens if t]
    if all(t in STOP_WORDS for t in content_tokens):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Text normalization
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str, **kwargs) -> str:
    """Normalize captions and write `text_normalized.parquet`."""
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]
    text_cfg = cfg["text"]

    in_path = os.path.join(intermediate_dir, "raw_unified.parquet")
    df = pd.read_parquet(in_path)
    before = len(df)
    log.info("Normalizing %d rows...", before)

    # Apply normalization
    df["caption"] = df["caption"].apply(
        lambda c: _normalize_caption(c, lowercase=text_cfg.get("lowercase", True))
    )

    # Filter by length and content
    min_w = text_cfg.get("min_word_count", 3)
    max_w = text_cfg.get("max_word_count", 50)
    mask = df["caption"].apply(lambda c: _is_valid(c, min_w, max_w))
    df = df[mask].reset_index(drop=True)

    after = len(df)
    log.info(
        "Text normalization complete: %d → %d rows (dropped %d, %.1f%%)",
        before, after, before - after, 100 * (before - after) / max(before, 1),
    )

    out_path = os.path.join(intermediate_dir, "text_normalized.parquet")
    df.to_parquet(out_path, index=False)
    log.info("Written: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def run_dedup(config_path: str, **kwargs) -> str:
    """Remove duplicate (video_id, caption) pairs and write `deduped.parquet`."""
    cfg = _load_config(config_path)
    intermediate_dir = cfg["paths"]["intermediate_dir"]

    in_path = os.path.join(intermediate_dir, "text_normalized.parquet")
    df = pd.read_parquet(in_path)
    before = len(df)

    # Intra-video deduplication: drop exact duplicate captions within same video
    df = df.drop_duplicates(subset=["video_id", "caption"]).reset_index(drop=True)
    after_intra = len(df)
    log.info(
        "Intra-video dedup: %d → %d rows (dropped %d)",
        before, after_intra, before - after_intra,
    )

    # Global deduplication: drop captions that are identical across videos
    # (keep one occurrence — retain the MSR-VTT version preferentially)
    df = df.sort_values("source")  # MSR-VTT sorts before MSVD alphabetically
    df = df.drop_duplicates(subset=["caption"], keep="first").reset_index(drop=True)
    after_global = len(df)
    log.info(
        "Global dedup: %d → %d rows (dropped %d)",
        after_intra, after_global, after_intra - after_global,
    )

    # Log per-source counts
    counts = df.groupby(["source", "split_hint"]).size().reset_index(name="count")
    log.info("Post-dedup counts:\n%s", counts.to_string(index=False))

    out_path = os.path.join(intermediate_dir, "deduped.parquet")
    df.to_parquet(out_path, index=False)
    log.info("Written: %s", out_path)
    return out_path
