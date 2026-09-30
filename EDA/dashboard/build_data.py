"""
build_data.py
─────────────
Compute the EDA statistics behind the dashboard and write them to `data.js`.

Sources (Hugging Face, same copies used by DATA/Download_Datasets.ipynb):
  - VLM2Vec/MSR-VTT : MSRVTT_data.json (10k videos, 200k captions),
                      msrvtt_train_9k.json / msrvtt_test_1k.json, raw_videos/*.mp4
  - VLM2Vec/MSVD    : msvd_{train,val,test}.json, raw_videos/*.avi

Caption statistics use every caption. Video properties (resolution, FPS,
measured duration, pixel statistics) are measured on a seeded random sample of
downloaded videos, controlled by --msrvtt-sample / --msvd-sample.

Usage:
    pip install -r EDA/dashboard/requirements.txt
    python EDA/dashboard/build_data.py --check-youtube
    open EDA/dashboard/index.html     # interactive; RESULTS.md has the same numbers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import nltk
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

HF = "https://huggingface.co/datasets"
MSRVTT_REPO = f"{HF}/VLM2Vec/MSR-VTT/resolve/main"
MSVD_REPO = f"{HF}/VLM2Vec/MSVD/resolve/main"
CLIP_TOKENIZER_URL = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json"

CLIP_MAX_TOKENS = 77
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Must match DATA/vlm_pipeline/config/pipeline_config.yaml `text:` section.
MIN_WORDS = 3
MAX_WORDS = 50

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch(url: str, dest: str, retries: int = 3) -> str | None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return dest
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def load_json(url: str, cache: str):
    path = fetch(url, os.path.join(cache, os.path.basename(url)))
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def hist(values, bins) -> dict:
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return {"edges": [round(float(e), 3) for e in edges], "counts": counts.tolist()}


def describe(values) -> dict:
    s = pd.Series(values, dtype=float)
    return {k: round(float(v), 3) for k, v in {
        "mean": s.mean(), "std": s.std(), "min": s.min(), "p25": s.quantile(.25),
        "median": s.median(), "p75": s.quantile(.75), "p95": s.quantile(.95), "max": s.max(),
    }.items()}


def duration_stats(durations: pd.Series, bins) -> dict:
    return {**describe(durations), "hist": hist(durations.clip(upper=60), bins=bins),
            "hist_1s": hist(durations.clip(upper=60), bins=range(0, 62))}


SPLIT_ORDER = {"train": 0, "train (9k)": 0, "val": 1, "test": 2, "test (1k-A)": 2}


def ordered(splits: list[dict]) -> list[dict]:
    return sorted(splits, key=lambda s: (s["scheme"], SPLIT_ORDER.get(s["split"], 9)))


def caption_stats(captions: pd.DataFrame, tokenizer: Tokenizer, stop_words: set[str]) -> dict:
    """`captions` has columns video_id, caption (one row per caption)."""
    text = captions["caption"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    words = text.str.split().str.len()
    enc = tokenizer.encode_batch(text.tolist(), add_special_tokens=False)
    tokens = np.array([len(e.ids) + 2 for e in enc])  # + <start>/<end>

    def all_stop(c: str) -> bool:
        toks = [t.strip(string.punctuation) for t in c.split()]
        toks = [t for t in toks if t]
        return all(t in stop_words for t in toks)

    too_short = words < MIN_WORDS
    too_long = words > MAX_WORDS
    only_stop = text.map(all_stop)
    keep = ~(too_short | too_long | only_stop)

    kept = pd.DataFrame({"video_id": captions["video_id"][keep], "caption": text[keep]})
    intra = kept.duplicated(["video_id", "caption"])
    kept = kept[~intra]
    cross = kept.duplicated("caption")

    content = Counter()
    bigrams = Counter()
    for c in text:
        toks = re.findall(r"[a-z']+", c)
        content.update(t for t in toks if t not in stop_words and len(t) > 1)
        bigrams.update(f"{a} {b}" for a, b in zip(toks, toks[1:])
                       if a not in stop_words and b not in stop_words)

    per_video = captions.groupby("video_id").size()
    n = len(captions)
    return {
        "total": n,
        "per_video": {**describe(per_video), "hist": hist(per_video, bins=min(40, int(per_video.max()) or 1))},
        "words": {**describe(words), "hist": hist(words.clip(upper=40), bins=range(0, 42))},
        "clip_tokens": {**describe(tokens), "hist": hist(np.clip(tokens, 0, 80), bins=range(0, 82))},
        "pct_over_77_tokens": round(100 * float((tokens > CLIP_MAX_TOKENS).mean()), 3),
        "vocab_size": len(content),
        "top_words": content.most_common(25),
        "top_bigrams": bigrams.most_common(20),
        "pipeline_filters": {
            "too_short": int(too_short.sum()),
            "too_long": int(too_long.sum()),
            "all_stopwords": int((only_stop & ~too_short & ~too_long).sum()),
            "intra_video_duplicates": int(intra.sum()),
            "cross_video_duplicates": int(cross.sum()),
            "retained": int((~cross).sum()),
            "retained_pct": round(100 * float((~cross).sum()) / n, 2),
        },
        "examples": {
            "shortest": text[words == words.min()].head(5).tolist(),
            "longest": text.loc[words.nlargest(3).index].tolist(),
            "most_repeated": text.value_counts().head(8).to_dict(),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Video probing
# ─────────────────────────────────────────────────────────────────────────────

def probe_video(path: str) -> dict | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "unknown"

    means, stds = [], []
    for frac in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n_frames * frac)))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        means.append(rgb.mean(axis=(0, 1)))
        stds.append(rgb.std(axis=(0, 1)))
    cap.release()
    if fps <= 0 or n_frames <= 0 or not means:
        return None
    return {
        "fps": round(fps, 3), "width": w, "height": h, "frames": n_frames,
        "duration": n_frames / fps, "codec": codec,
        "rgb_mean": np.mean(means, axis=0).tolist(), "rgb_std": np.mean(stds, axis=0).tolist(),
        "size_mb": os.path.getsize(path) / 1e6,
    }


def aspect_label(w: int, h: int) -> str:
    r = w / h
    for name, target in (("4:3", 4 / 3), ("16:9", 16 / 9), ("3:2", 1.5), ("1:1", 1.0), ("5:4", 1.25)):
        if abs(r - target) < 0.03:
            return name
    return "other"


def video_stats(repo: str, files: list[str], cache: str, workers: int) -> dict:
    def work(fname):
        path = fetch(f"{repo}/raw_videos/{fname}", os.path.join(cache, fname))
        return probe_video(path) if path else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        probed = [p for p in ex.map(work, files) if p]

    df = pd.DataFrame(probed)
    res = (df["width"].astype(str) + "x" + df["height"].astype(str)).value_counts()
    aspect = df.apply(lambda r: aspect_label(r.width, r.height), axis=1).value_counts()
    rgb_mean = np.mean(df["rgb_mean"].tolist(), axis=0)
    rgb_std = np.mean(df["rgb_std"].tolist(), axis=0)
    frames_1fps = np.floor(df["duration"]).clip(lower=1)
    return {
        "requested": len(files),
        "probed": len(df),
        "fps": {**describe(df["fps"]), "counts": df["fps"].round(0).astype(int).value_counts().sort_index().to_dict()},
        "duration": {**describe(df["duration"]), "hist": hist(df["duration"].clip(upper=60), bins=30)},
        "resolutions": res.head(12).to_dict(),
        "n_resolutions": int(res.size),
        "aspect_ratios": aspect.to_dict(),
        "width": describe(df["width"]),
        "height": describe(df["height"]),
        "codecs": df["codec"].value_counts().to_dict(),
        "file_size_mb": describe(df["size_mb"]),
        "frames_per_video_at_1fps": describe(frames_1fps),
        "frames_per_video_native": describe(df["frames"]),
        "rgb_mean": [round(float(x), 4) for x in rgb_mean],
        "rgb_std": [round(float(x), 4) for x in rgb_std],
        "brightness_hist": hist([np.mean(m) for m in df["rgb_mean"]], bins=np.linspace(0, 1, 21)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# YouTube availability
# ─────────────────────────────────────────────────────────────────────────────

def youtube_status(video_ids: list[str], cache: str) -> dict:
    """oEmbed status per YouTube id. 200/401 = live, 403 = private/removed, 404 = deleted."""
    cache_path = os.path.join(cache, "youtube_status.json")
    known = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def check(yt_id):
        url = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(
            f"https://www.youtube.com/watch?v={yt_id}", safe="")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    return yt_id, r.status
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(5 * 2 ** attempt)
                    continue
                return yt_id, e.code
            except Exception:
                time.sleep(2)
        return yt_id, 0

    todo = [v for v in set(video_ids) if v not in known]
    with ThreadPoolExecutor(max_workers=16) as ex:
        known.update(dict(ex.map(check, todo)))
    json.dump(known, open(cache_path, "w"))
    return known


def availability_summary(clip_to_yt: dict[str, str], status: dict, split_of: dict[str, str]) -> dict:
    live = {k: status.get(v) in (200, 401) for k, v in clip_to_yt.items()}
    by_split = Counter()
    tot_split = Counter()
    for clip, ok in live.items():
        tot_split[split_of.get(clip, "all")] += 1
        by_split[split_of.get(clip, "all")] += ok
    codes = Counter(status.get(v) for v in set(clip_to_yt.values()))
    return {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "unique_sources": len(set(clip_to_yt.values())),
        "status_counts": {str(k): v for k, v in codes.items()},
        "clips_available": int(sum(live.values())),
        "clips_total": len(live),
        "by_split": {s: {"available": int(by_split[s]), "total": int(tot_split[s])} for s in tot_split},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_msrvtt(args, tokenizer, stop_words) -> dict:
    cache = os.path.join(args.cache, "msrvtt")
    data = load_json(f"{MSRVTT_REPO}/raw_data/MSRVTT_data.json", cache)
    train9k = load_json(f"{MSRVTT_REPO}/msrvtt_train_9k.json", cache)
    test1k = load_json(f"{MSRVTT_REPO}/msrvtt_test_1k.json", cache)
    cat_names = [line.split("\t")[0] for line in
                 open(fetch(f"{MSRVTT_REPO}/raw_data/category.txt", os.path.join(cache, "category.txt")))
                 if line.strip()]

    videos = pd.DataFrame(data["videos"])
    videos["duration"] = videos["end time"] - videos["start time"]
    videos["category_name"] = videos["category"].map(lambda c: cat_names[c])
    # This copy of MSRVTT_data.json labels every video "train"; the official
    # MSR-VTT split is by id: 0-6512 train, 6513-7009 val, 7010-9999 test.
    vid_num = videos["video_id"].str.replace("video", "").astype(int)
    videos["official_split"] = np.select([vid_num < 6513, vid_num < 7010], ["train", "val"], "test")
    test_ids = {v["video_id"] for v in test1k}
    train_ids = {v["video_id"] for v in train9k}
    videos["retrieval_split"] = np.where(videos.video_id.isin(test_ids), "test (1k-A)",
                                         np.where(videos.video_id.isin(train_ids), "train (9k)", "unused"))
    sentences = pd.DataFrame(data["sentences"])

    splits = []
    for scheme, col in (("Official (train/val/test)", "official_split"), ("Retrieval (9k / 1k-A)", "retrieval_split")):
        caps = sentences.merge(videos[["video_id", col]], on="video_id").groupby(col).size()
        for split, n in videos[col].value_counts().items():
            splits.append({"scheme": scheme, "split": split, "videos": int(n), "captions": int(caps.get(split, 0))})

    cats = (videos.groupby(["category_name", "official_split"]).size().unstack(fill_value=0)
            .reindex(columns=["train", "val", "test"], fill_value=0))
    cats = cats.loc[cats.sum(axis=1).sort_values(ascending=False).index]

    rng = random.Random(args.seed)
    sample_ids = rng.sample(sorted(videos.video_id), min(args.msrvtt_sample, len(videos)))
    print(f"[MSR-VTT] probing {len(sample_ids)} videos...")
    vstats = video_stats(MSRVTT_REPO, [f"{v}.mp4" for v in sample_ids], cache, args.workers)

    print("[MSR-VTT] caption statistics...")
    cstats = caption_stats(sentences[["video_id", "caption"]], tokenizer, stop_words)

    out = {
        "name": "MSR-VTT",
        "source": "https://huggingface.co/datasets/VLM2Vec/MSR-VTT",
        "videos": len(videos),
        "splits": ordered(splits),
        "categories": {"names": cats.index.tolist(), **{s: cats[s].tolist() for s in cats.columns}},
        "annotated_duration": duration_stats(videos["duration"], bins=40),
        "video": vstats,
        "captions": cstats,
        "source_videos": int(videos["url"].nunique()),
    }
    if args.check_youtube:
        print("[MSR-VTT] checking YouTube availability...")
        clip_to_yt = {r.video_id: r.url.split("v=")[-1] for r in videos.itertuples()}
        status = youtube_status(list(clip_to_yt.values()), args.cache)
        out["youtube"] = availability_summary(clip_to_yt, status, dict(zip(videos.video_id, videos.official_split)))
    return out


def build_msvd(args, tokenizer, stop_words) -> dict:
    cache = os.path.join(args.cache, "msvd")
    rows, split_of = [], {}
    for split, fname in (("train", "msvd_train.json"), ("val", "msvd_val.json"), ("test", "msvd_test.json")):
        for v in load_json(f"{MSVD_REPO}/{fname}", cache):
            split_of[v["video_id"]] = split
            rows.append({"video_id": v["video_id"], "file": v["video"], "split": split,
                         "n_caps": len(v["caption"]), "_caps": v["caption"]})
    videos = pd.DataFrame(rows)
    # MSVD ids are <youtube_id>_<start>_<end>
    parts = videos.video_id.str.rsplit("_", n=2, expand=True)
    videos["yt_id"] = parts[0]
    videos["duration"] = parts[2].astype(float) - parts[1].astype(float)
    sentences = videos[["video_id", "_caps"]].explode("_caps").rename(columns={"_caps": "caption"}).dropna()

    splits = [{"scheme": "Official", "split": s, "videos": int(g.shape[0]), "captions": int(g.n_caps.sum())}
              for s, g in videos.groupby("split")]

    rng = random.Random(args.seed)
    sample = rng.sample(sorted(videos.file), min(args.msvd_sample, len(videos)))
    print(f"[MSVD] probing {len(sample)} videos...")
    vstats = video_stats(MSVD_REPO, sample, cache, args.workers)

    print("[MSVD] caption statistics...")
    cstats = caption_stats(sentences, tokenizer, stop_words)

    out = {
        "name": "MSVD",
        "source": "https://huggingface.co/datasets/VLM2Vec/MSVD",
        "videos": len(videos),
        "splits": ordered(splits),
        "categories": None,
        "annotated_duration": duration_stats(videos["duration"], bins=30),
        "video": vstats,
        "captions": cstats,
        "source_videos": int(videos["yt_id"].nunique()),
    }
    if args.check_youtube:
        print("[MSVD] checking YouTube availability...")
        clip_to_yt = dict(zip(videos.video_id, videos.yt_id))
        status = youtube_status(list(clip_to_yt.values()), args.cache)
        out["youtube"] = availability_summary(clip_to_yt, status, split_of)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _n(x, d=0):
    return f"{x:,.{d}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    s = payload["settings"]
    ds = payload["datasets"]
    clip_mean = payload["clip_norm"]["mean"]
    out = [
        "# EDA Results: MSR-VTT and MSVD",
        "",
        f"Generated {payload['generated_at']} by `EDA/dashboard/build_data.py`. Do not edit by hand; "
        "re-run the script to refresh. Open `index.html` for the interactive version.",
        "",
        f"Caption statistics use every caption. Video properties are measured on a random sample "
        f"(seed {s['seed']}) of {_n(s['msrvtt_sample'])} MSR-VTT and {_n(s['msvd_sample'])} MSVD videos.",
        "",
        "![MSR-VTT view](screenshots/msrvtt.png)",
        "",
        "## Overview",
        "",
    ]

    def row(label, f):
        return [label] + [f(d) for d in ds.values()]

    def yt(d):
        y = d.get("youtube")
        return f"{100 * y['clips_available'] / y['clips_total']:.1f}%" if y else "not checked"

    def top_res(d):
        res, n = next(iter(d["video"]["resolutions"].items()))
        return f"{res} ({100 * n / d['video']['probed']:.0f}%)"

    out.append(_table(["Metric"] + [d["name"] for d in ds.values()], [
        row("Videos", lambda d: _n(d["videos"])),
        row("Source YouTube videos", lambda d: _n(d["source_videos"])),
        row("Captions", lambda d: _n(d["captions"]["total"])),
        row("Captions per video (mean / min / max)", lambda d: "{} / {} / {}".format(
            _n(d["captions"]["per_video"]["mean"], 1), _n(d["captions"]["per_video"]["min"]),
            _n(d["captions"]["per_video"]["max"]))),
        row("Duration mean / median (s)", lambda d: f"{d['annotated_duration']['mean']:.1f} / {d['annotated_duration']['median']:.1f}"),
        row("Duration range (s)", lambda d: f"{d['annotated_duration']['min']:.1f} – {d['annotated_duration']['max']:.1f}"),
        row("FPS mean (sample)", lambda d: f"{d['video']['fps']['mean']:.1f}"),
        row("Distinct resolutions (sample)", lambda d: _n(d["video"]["n_resolutions"])),
        row("Top resolution (sample)", top_res),
        row("Words per caption mean / max", lambda d: f"{d['captions']['words']['mean']:.1f} / {_n(d['captions']['words']['max'])}"),
        row("CLIP tokens p95 / max", lambda d: f"{_n(d['captions']['clip_tokens']['p95'])} / {_n(d['captions']['clip_tokens']['max'])}"),
        row("Captions > 77 CLIP tokens", lambda d: f"{d['captions']['pct_over_77_tokens']:.2f}%"),
        row("Vocabulary (content words)", lambda d: _n(d["captions"]["vocab_size"])),
        row("Captions kept by pipeline text rules", lambda d: f"{d['captions']['pipeline_filters']['retained_pct']:.1f}%"),
        row("Clips still on YouTube", yt),
    ]))

    for d in ds.values():
        c, v, f = d["captions"], d["video"], d["captions"]["pipeline_filters"]
        out += ["", f"## {d['name']}", "", f"Source: <{d['source']}>", "", "### Splits", ""]
        out.append(_table(["Scheme", "Split", "Videos", "Captions"],
                          [[x["scheme"], x["split"], _n(x["videos"]), _n(x["captions"])] for x in d["splits"]]))
        if d["categories"]:
            cat = d["categories"]
            out += ["", "### Categories (official split)", ""]
            out.append(_table(["Category", "Train", "Val", "Test", "Total"], [
                [n, _n(tr), _n(va), _n(te), _n(tr + va + te)]
                for n, tr, va, te in zip(cat["names"], cat["train"], cat["val"], cat["test"])]))
        out += ["", "### Video properties (sample)", ""]
        out.append(_table(["Property", "Value"], [
            ["Videos probed", f"{_n(v['probed'])} of {_n(v['requested'])}"],
            ["FPS mean / median / min / max", f"{v['fps']['mean']:.1f} / {v['fps']['median']:.1f} / {v['fps']['min']:.1f} / {v['fps']['max']:.1f}"],
            ["Measured duration median (s)", f"{v['duration']['median']:.1f}"],
            ["Native frames per video (median)", _n(v["frames_per_video_native"]["median"])],
            ["Frames per video at 1 fps (median)", _n(v["frames_per_video_at_1fps"]["median"])],
            ["Resolutions", ", ".join(f"{k} ({n})" for k, n in list(v["resolutions"].items())[:6])],
            ["Aspect ratios", ", ".join(f"{k} ({100 * n / v['probed']:.0f}%)" for k, n in v["aspect_ratios"].items())],
            ["Codecs", ", ".join(f"{k} ({n})" for k, n in v["codecs"].items())],
            ["File size median (MB)", f"{v['file_size_mb']['median']:.2f}"],
            ["RGB mean (0–1)", ", ".join(f"{x:.3f}" for x in v["rgb_mean"])],
            ["RGB std (0–1)", ", ".join(f"{x:.3f}" for x in v["rgb_std"])],
            ["CLIP normalization mean", ", ".join(f"{x:.3f}" for x in clip_mean)],
        ]))
        out += ["", "### Captions", ""]
        out.append(_table(["Statistic", "Mean", "Median", "p95", "Max"], [
            ["Words per caption", f"{c['words']['mean']:.1f}", _n(c["words"]["median"]), _n(c["words"]["p95"]), _n(c["words"]["max"])],
            ["CLIP tokens per caption", f"{c['clip_tokens']['mean']:.1f}", _n(c["clip_tokens"]["median"]), _n(c["clip_tokens"]["p95"]), _n(c["clip_tokens"]["max"])],
            ["Captions per video", f"{c['per_video']['mean']:.1f}", _n(c["per_video"]["median"]), _n(c["per_video"]["p95"]), _n(c["per_video"]["max"])],
        ]))
        out += ["", "Top 15 content words: " + ", ".join(f"{w} ({_n(n)})" for w, n in c["top_words"][:15]) + ".",
                "", "Top 10 bigrams: " + ", ".join(f"{w} ({_n(n)})" for w, n in c["top_bigrams"][:10]) + "."]
        out += ["", "### Pipeline text filters", "",
                f"Applying the rules in `DATA/vlm_pipeline/tasks/text_normalize.py` (min {s['min_words']} words, "
                f"max {s['max_words']} words, no all-stopword captions, then dedup):", ""]
        out.append(_table(["Step", "Captions removed"], [
            [f"Fewer than {s['min_words']} words", _n(f["too_short"])],
            [f"More than {s['max_words']} words", _n(f["too_long"])],
            ["All stopwords", _n(f["all_stopwords"])],
            ["Duplicate within the same video", _n(f["intra_video_duplicates"])],
            ["Duplicate across videos (global dedup)", _n(f["cross_video_duplicates"])],
            ["**Retained**", f"**{_n(f['retained'])} ({f['retained_pct']:.1f}%)**"],
        ]))
        out += ["", "Most repeated captions: " + "; ".join(
            f"\"{k}\" ({_n(n)})" for k, n in list(c["examples"]["most_repeated"].items())[:6]) + "."]
        if d.get("youtube"):
            y = d["youtube"]
            out += ["", f"### YouTube availability (checked {y['checked_at']})", ""]
            out.append(_table(["Split", "Available", "Total", "Share"], [
                [k, _n(x["available"]), _n(x["total"]), f"{100 * x['available'] / x['total']:.1f}%"]
                for k, x in y["by_split"].items()]))
            codes = y["status_counts"]
            out += ["", f"Unique source videos: {_n(y['unique_sources'])}. oEmbed status: "
                    f"live {_n(codes.get('200', 0))}, embedding disabled {_n(codes.get('401', 0))}, "
                    f"private/removed {_n(codes.get('403', 0))}, deleted {_n(codes.get('404', 0))}."]

    m, v = ds["msrvtt"], ds["msvd"]
    mf, vf = m["captions"]["pipeline_filters"], v["captions"]["pipeline_filters"]
    top_m = next(iter(m["video"]["resolutions"]))
    out += [
        "", "## What this means for the pipeline", "",
        f"- **Caption length is not a constraint.** {m['captions']['pct_over_77_tokens']:.2f}% of MSR-VTT and "
        f"{v['captions']['pct_over_77_tokens']:.2f}% of MSVD captions exceed CLIP's 77 tokens "
        f"(max {_n(m['captions']['clip_tokens']['max'])} and {_n(v['captions']['clip_tokens']['max'])}).",
        f"- **The HF MSR-VTT copy is re-encoded** to {top_m} at {m['video']['fps']['mean']:.0f} fps in the sample, "
        "so frames cannot be sampled faster than that. MSVD keeps its original, varied resolutions "
        f"({_n(v['video']['n_resolutions'])} distinct in the sample) and ~{v['video']['fps']['median']:.0f} fps.",
        "- **Resize the short side to 224 and center-crop** instead of squashing frames to 224×224; most clips are 4:3.",
        "- **Use CLIP's normalization constants.** Dataset pixel means are below CLIP's "
        f"(MSR-VTT {', '.join(f'{x:.3f}' for x in m['video']['rgb_mean'])} vs CLIP {', '.join(f'{x:.3f}' for x in clip_mean)}).",
        f"- **Global dedup drops valid captions:** {_n(mf['cross_video_duplicates'])} MSR-VTT and "
        f"{_n(vf['cross_video_duplicates'])} MSVD captions are removed only because another video has the same text. "
        "Consider keeping within-video dedup only.",
        f"- **MSVD caption counts vary** from {_n(v['captions']['per_video']['min'])} to "
        f"{_n(v['captions']['per_video']['max'])} per video; sample a fixed number per video per epoch.",
    ]
    if m.get("youtube") and v.get("youtube"):
        out.append(f"- **Use the Hugging Face copies, not YouTube URLs:** only {yt(m)} of MSR-VTT and {yt(v)} of MSVD "
                   "clips are still online.")
    out += ["", "## Screenshots", "", "![MSVD view](screenshots/msvd.png)", "", "![Compare view](screenshots/compare.png)", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/vlm_eda"), help="download cache")
    ap.add_argument("--msrvtt-sample", type=int, default=1000, help="MSR-VTT videos to probe")
    ap.add_argument("--msvd-sample", type=int, default=400, help="MSVD videos to probe")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-youtube", action="store_true", help="check source-video availability on YouTube")
    ap.add_argument("--out", default=os.path.join(HERE, "data.js"))
    args = ap.parse_args()

    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))

    tokenizer = Tokenizer.from_file(fetch(CLIP_TOKENIZER_URL, os.path.join(args.cache, "clip_tokenizer.json")))

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": {"msrvtt_sample": args.msrvtt_sample, "msvd_sample": args.msvd_sample, "seed": args.seed,
                     "min_words": MIN_WORDS, "max_words": MAX_WORDS, "clip_max_tokens": CLIP_MAX_TOKENS},
        "clip_norm": {"mean": CLIP_MEAN, "std": CLIP_STD},
        "datasets": {"msrvtt": build_msrvtt(args, tokenizer, stop_words),
                     "msvd": build_msvd(args, tokenizer, stop_words)},
    }
    # A .js file (not .json) so index.html works when opened directly from disk.
    with open(args.out, "w") as f:
        f.write("window.EDA_DATA = ")
        json.dump(payload, f, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write(";\n")
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")

    md_path = os.path.join(os.path.dirname(args.out), "RESULTS.md")
    with open(md_path, "w") as f:
        f.write(render_markdown(json.loads(json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)))))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
