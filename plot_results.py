from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ORDER = ["offline", "naive", "replay", "ewc", "proposed"]


def load_metrics(root: Path):
    frames = []
    for path in root.glob("metrics_*.csv"):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No metrics_*.csv files found in %s" % root)
    return pd.concat(frames, ignore_index=True)


def line_plot(df, dataset, metric, ylabel, path):
    subset = df[(df.dataset == dataset) & (df.batch >= 0)]
    stats = subset.groupby(["method", "batch"])[metric].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in ORDER:
        m = stats[stats.method == method]
        if m.empty:
            continue
        ax.plot(m.batch, m["mean"], label=method)
        if m["std"].notna().any():
            ax.fill_between(m.batch, m["mean"] - m["std"].fillna(0), m["mean"] + m["std"].fillna(0), alpha=0.15)
    ax.set_xlabel("Stream batch")
    ax.set_ylabel(ylabel)
    ax.set_title("%s: %s over the stream" % (dataset.upper(), ylabel))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/extended")
    args = p.parse_args()
    root = Path(args.results_dir)
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(root)
    for dataset in sorted(df.dataset.unique()):
        line_plot(df, dataset, "test_accuracy", "Test accuracy", figdir / ("%s_accuracy.png" % dataset))
        line_plot(df, dataset, "pseudo_precision", "Pseudo-label precision", figdir / ("%s_pseudo_precision.png" % dataset))
        line_plot(df, dataset, "pseudo_coverage", "Pseudo-label coverage", figdir / ("%s_pseudo_coverage.png" % dataset))
        line_plot(df, dataset, "parameter_change", "Relative parameter change", figdir / ("%s_parameter_change.png" % dataset))
        line_plot(df, dataset, "feature_centroid_drift", "Feature-centroid drift", figdir / ("%s_feature_drift.png" % dataset))
    print("Figures saved to", figdir)


if __name__ == "__main__":
    main()
