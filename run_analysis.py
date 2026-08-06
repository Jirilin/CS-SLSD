from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def safe_corr(a, b):
    data = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(data) < 3 or data.a.nunique() < 2 or data.b.nunique() < 2:
        return float("nan")
    return float(data.a.corr(data.b))

def main():
    p = argparse.ArgumentParser(description="Analyse why continual-learning performance changes")
    p.add_argument("--results-dir", default="results/repeated")
    args = p.parse_args()
    root = Path(args.results_dir)
    df = pd.read_csv(root / "all_runs.csv")
    stream = df[df.batch >= 0].copy()
    records = []
    for (method, seed), g in stream.groupby(["method", "seed"]):
        g = g.sort_values("batch")
        records.append({
            "method": method,
            "seed": seed,
            "corr_distribution_vs_accuracy": safe_corr(g.distribution_tv, g.test_accuracy),
            "corr_distribution_vs_precision": safe_corr(g.distribution_tv, g.pseudo_precision),
            "corr_parameter_change_vs_accuracy": safe_corr(g.parameter_change, g.test_accuracy),
            "corr_centroid_drift_vs_accuracy": safe_corr(g.feature_centroid_drift, g.test_accuracy),
            "accuracy_slope": float(np.polyfit(g.batch, g.test_accuracy, 1)[0]) if len(g) >= 2 else np.nan,
        })
    per_run = pd.DataFrame(records)
    per_run.to_csv(root / "change_correlation_per_run.csv", index=False)
    summary = per_run.groupby("method", as_index=False).agg(
        corr_distribution_accuracy_mean=("corr_distribution_vs_accuracy", "mean"),
        corr_distribution_accuracy_std=("corr_distribution_vs_accuracy", "std"),
        corr_distribution_precision_mean=("corr_distribution_vs_precision", "mean"),
        corr_parameter_accuracy_mean=("corr_parameter_change_vs_accuracy", "mean"),
        corr_centroid_accuracy_mean=("corr_centroid_drift_vs_accuracy", "mean"),
        accuracy_slope_mean=("accuracy_slope", "mean"),
    )
    summary.to_csv(root / "change_correlation_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
