
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    ("final_accuracy_mean", "Final accuracy"),
    ("classwise_forgetting_mean", "Class-wise forgetting"),
    ("mean_pseudo_precision_mean", "Pseudo-label precision"),
    ("mean_pseudo_coverage_mean", "Pseudo-label coverage"),
    ("mean_batch_seconds_mean", "Mean seconds per stream batch"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/extended")
    args = p.parse_args()
    root = Path(args.results_dir)
    src = root / "comparison_mean_std.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Run aggregate_results.py first.")

    df = pd.read_csv(src)
    out_fig = Path("results/final_figures")
    out_tab = Path("results/final_tables")
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    # Keep a compact dissertation-facing table.
    desired = ["dataset", "method"]
    for metric, _ in METRICS:
        if metric in df.columns:
            desired.append(metric)
        std_col = metric.replace("_mean", "_std")
        if std_col in df.columns:
            desired.append(std_col)
    compact = df[[c for c in desired if c in df.columns]].copy()
    compact.to_csv(out_tab / "final_comparison_table.csv", index=False)

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].copy()
        for metric, label in METRICS:
            if metric not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.bar(sub["method"], sub[metric])
            ax.set_title(f"{dataset.upper()} — {label}")
            ax.set_ylabel(label)
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            fig.savefig(out_fig / f"{dataset}_{metric}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved final table to {out_tab / 'final_comparison_table.csv'}")
    print(f"Saved figures to {out_fig}")


if __name__ == "__main__":
    main()
