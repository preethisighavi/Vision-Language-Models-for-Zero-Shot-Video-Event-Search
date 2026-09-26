"""
Shared results-table harness.

Every model evaluation notebook should call log_metrics() instead of writing
its own standalone results JSON, so all models/datasets end up in one
comparable table: results/model_results.csv

Usage (from a notebook, after computing a metrics dict):

    from results.log_result import log_metrics

    log_metrics(
        model="CLIP",
        variant="fine_tuned",
        dataset="MSVD",
        metrics={"R@1": 0.366, "R@5": 0.679, "MRR": 0.507},
        source_file="clip_finetuned_results.json",
    )
"""

import csv
import os
from typing import Dict

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_results.csv")
FIELDNAMES = ["model", "variant", "dataset", "metric_name", "metric_value", "source_file"]


def log_metrics(model: str, variant: str, dataset: str, metrics: Dict[str, float], source_file: str) -> None:
    """Append one row per metric to the shared results table."""
    file_exists = os.path.isfile(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for metric_name, metric_value in metrics.items():
            writer.writerow({
                "model": model,
                "variant": variant,
                "dataset": dataset,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "source_file": source_file,
            })

    print(f"Logged {len(metrics)} metrics for {model}/{variant}/{dataset} to {RESULTS_CSV}")


def load_results():
    """Read the shared results table back as a list of dict rows."""
    if not os.path.isfile(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
