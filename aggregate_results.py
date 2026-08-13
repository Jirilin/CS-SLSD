from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


KEY_METRICS = [
    "final_accuracy", "mean_stream_accuracy", "classwise_forgetting",
    "average_incremental_accuracy", "backward_transfer_proxy",
    "mean_pseudo_precision", "mean_pseudo_coverage",
    "mean_feature_centroid_drift", "mean_parameter_change",
    "mean_batch_seconds", "total_seconds",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/extended")
    args = p.parse_args()
    root = Path(args.results_dir)
    summaries = []
    for path in sorted(root.glob("summary_*.json")):
        with open(path) as f:
            summaries.append(json.load(f))
    if not summaries:
        raise SystemExit("No summary_*.json files found in %s" % root)
    raw = pd.DataFrame(summaries)
    raw.to_csv(root / "all_run_summaries.csv", index=False)

    rows = []
    for (dataset, method), group in raw.groupby(["dataset", "method"], sort=True):
        row = {"dataset": dataset, "method": method, "runs": len(group)}
        for metric in KEY_METRICS:
            if metric in group:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(root / "comparison_mean_std.csv", index=False)

    # Compact dissertation-ready table.
    compact = table[[
        "dataset", "method", "runs",
        "final_accuracy_mean", "final_accuracy_std",
        "classwise_forgetting_mean", "classwise_forgetting_std",
        "mean_pseudo_precision_mean", "mean_pseudo_coverage_mean",
        "mean_batch_seconds_mean",
    ]].copy()
    compact.to_csv(root / "dissertation_comparison_table.csv", index=False)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
