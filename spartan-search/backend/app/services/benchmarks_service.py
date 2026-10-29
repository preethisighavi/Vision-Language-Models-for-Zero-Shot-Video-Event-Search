"""Reads results/model_results.csv (read-only) and returns it in long format.

The CSV has columns model, variant, dataset, metric_name, metric_value, source_file.
Optional `protocol` and `notes` columns are used when present; otherwise they
come from benchmarks_meta.yaml, matched on source_file or model/variant/dataset.
"""

import csv
from pathlib import Path

import yaml

from app.config import get_settings
from app.schemas.api import BenchmarkRow

METRIC_NAMES = {
    "R@1": "R@1", "R@5": "R@5", "R@10": "R@10", "MRR": "MRR",
    "MedianRank": "MedR", "MedR": "MedR", "MeanRank": "MeanR", "MeanR": "MeanR",
    "R1@0.5": "R1@0.5", "R1@0.7": "R1@0.7", "mIoU": "mIoU",
}
DATASET_NAMES = {"MSVD": "MSVD", "MSRVTT": "MSR-VTT", "MSR-VTT": "MSR-VTT"}


def _model_label(model: str, variant: str) -> str:
    variant = (variant or "").replace("_", "-").strip()
    return f"{model} ({variant})" if variant else model


def _load_meta(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("rows", [])


def _lookup(meta: list[dict], row: dict) -> dict:
    for m in meta:
        match = m.get("match", {})
        if all(str(row.get(k, "")).lower() == str(v).lower() for k, v in match.items()):
            return m
    return {}


def load_benchmarks(dataset: str | None = None, model: str | None = None) -> list[BenchmarkRow]:
    settings = get_settings()
    path = Path(settings.results_csv)
    if not path.exists():
        return []
    meta = _load_meta(Path(settings.benchmarks_meta))
    rows: list[BenchmarkRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = raw.get("metric_name", "")
            if name == "avg_query_latency_sec":
                metric, value = "latency_ms", float(raw["metric_value"]) * 1000.0
            elif name in METRIC_NAMES:
                metric, value = METRIC_NAMES[name], float(raw["metric_value"])
            else:
                continue  # Top-1/Top-5 accuracy duplicate R@1/R@5
            info = _lookup(meta, raw)
            rows.append(BenchmarkRow(
                model=_model_label(raw.get("model", ""), raw.get("variant", "")),
                dataset=DATASET_NAMES.get(raw.get("dataset", ""), raw.get("dataset", "")),
                protocol=raw.get("protocol") or info.get("protocol", ""),
                metric=metric,
                value=round(value, 6),
                notes=raw.get("notes") or info.get("notes", ""),
            ))
    if dataset:
        rows = [r for r in rows if r.dataset.lower() == dataset.lower()]
    if model:
        rows = [r for r in rows if model.lower() in r.model.lower()]
    return rows
