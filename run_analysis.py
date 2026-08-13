from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def safe_corr(group, a, b):
    x = group[[a, b]].dropna()
    if len(x) < 3 or x[a].nunique() < 2 or x[b].nunique() < 2:
        return float("nan")
    return float(x[a].corr(x[b]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/extended")
    args = p.parse_args()
    root = Path(args.results_dir)
    frames = [pd.read_csv(p) for p in root.glob("metrics_*.csv")]
    if not frames:
        raise SystemExit("No metrics files found")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.batch >= 0]
    rows = []
    for (dataset, method, seed), group in df.groupby(["dataset", "method", "seed"]):
        rows.append({
            "dataset": dataset, "method": method, "seed": seed,
            "distribution_vs_accuracy": safe_corr(group, "distribution_tv", "test_accuracy"),
            "distribution_vs_precision": safe_corr(group, "distribution_tv", "pseudo_precision"),
            "feature_drift_vs_accuracy": safe_corr(group, "feature_centroid_drift", "test_accuracy"),
            "parameter_change_vs_accuracy": safe_corr(group, "parameter_change", "test_accuracy"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(root / "change_correlation_per_run.csv", index=False)
    summary = out.groupby(["dataset", "method"]).agg(["mean", "std"]).reset_index()
    summary.to_csv(root / "change_correlation_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
