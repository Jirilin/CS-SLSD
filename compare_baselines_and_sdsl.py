from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["offline", "naive", "replay", "ewc", "sdsl_full"]


def load_jsons(root: Path):
    rows = []
    for path in sorted(root.glob("summary_*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            d.pop("config", None)
            if "dataset" in d and "method" in d:
                d["_source_file"] = path.name
                rows.append(d)
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", default="results/baselines")
    p.add_argument("--sdsl-dir", default="results/sdsl_full")
    p.add_argument("--output", default="results/final_method_comparison.csv")
    p.add_argument(
        "--raw-output",
        default="results/final_method_runs.csv",
    )
    args = p.parse_args()

    rows = load_jsons(Path(args.baseline_dir)) + load_jsons(Path(args.sdsl_dir))
    if not rows:
        raise SystemExit("No summary JSON files found.")

    df = pd.DataFrame(rows)
    Path(args.raw_output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.raw_output, index=False)

    candidate_metrics = [
        "initial_accuracy",
        "final_accuracy",
        "mean_stream_accuracy",
        "classwise_forgetting",
        "average_incremental_accuracy",
        "backward_transfer_proxy",
        "mean_pseudo_precision",
        "mean_pseudo_coverage",
        "mean_parameter_change",
        "mean_batch_seconds",
        "total_seconds",
    ]
    metrics = [c for c in candidate_metrics if c in df.columns]

    grouped = (
        df.groupby(["dataset", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = ["_".join(x) for x in grouped.columns]
    grouped = grouped.reset_index()

    grouped["method"] = pd.Categorical(
        grouped["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    grouped = grouped.sort_values(["dataset", "method"]).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.output, index=False)

    print(grouped.to_string(index=False))
    print(
        "\nInterpretation rule:"
        "\n  • Offline is an oracle joint-supervised reference and has access to labels"
        " unavailable to the continual semi-supervised methods."
        "\n  • Do not compare Offline stream-trajectory/forgetting metrics when they are NaN."
        "\n  • Compare Naive, Replay, EWC and SDSL under matched dataset/seed conditions."
        "\n  • Only claim SDSL superiority when measured means, variability and runtime"
        " support that claim."
    )


if __name__ == "__main__":
    main()
