from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def load_jsons(root: Path):
    rows = []
    for path in root.glob("summary_*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            d.pop("config", None)
            if "dataset" in d and "method" in d:
                rows.append(d)
        except Exception:
            continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", default="results/extended")
    p.add_argument("--sdsl-dir", default="results/sdsl_full")
    p.add_argument("--output", default="results/final_method_comparison.csv")
    args = p.parse_args()
    rows = load_jsons(Path(args.baseline_dir)) + load_jsons(Path(args.sdsl_dir))
    if not rows:
        raise SystemExit("No summary JSON files found")
    df = pd.DataFrame(rows)
    metrics = [c for c in ["final_accuracy", "mean_stream_accuracy", "classwise_forgetting",
                           "mean_pseudo_precision", "mean_pseudo_coverage", "mean_batch_seconds"]
               if c in df.columns]
    grouped = df.groupby(["dataset", "method"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(x) for x in grouped.columns]
    grouped = grouped.reset_index()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.output, index=False)
    print(grouped.to_string(index=False))
    print("\nEvidence rule: only claim SDSL is better where the measured mean improves and variability/compute trade-offs are reported.")


if __name__ == "__main__":
    main()
