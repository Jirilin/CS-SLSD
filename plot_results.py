from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PREFERRED_ORDER = ["offline", "naive", "replay", "ewc", "proposed"]


def normalise_method_name(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "naive_online": "naive",
        "naive online": "naive",
        "baseline_naive": "naive",
        "baseline_naive_online": "naive",
        "offline_only": "offline",
        "offline-only": "offline",
        "baseline_offline": "offline",
        "centroid_replay": "replay",
        "replay_buffer": "replay",
        "baseline_replay": "replay",
        "centroid_ewc": "ewc",
        "online_ewc": "ewc",
        "centroid_replay_ewc": "proposed",
        "combined": "proposed",
        "proposed_method": "proposed",
    }
    return aliases.get(value, value)


def load_metrics(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("metrics_*.csv"))
    if not paths:
        raise SystemExit(f"No metrics_*.csv files found in {root}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    required = {"dataset", "method", "batch"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            "Metrics files are missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["method_original"] = df["method"].astype(str)
    df["method"] = df["method"].map(normalise_method_name)
    df["batch"] = pd.to_numeric(df["batch"], errors="coerce")
    return df


def ordered_methods(df: pd.DataFrame) -> list[str]:
    found = [str(x) for x in df["method"].dropna().unique()]
    preferred = [m for m in PREFERRED_ORDER if m in found]
    extras = sorted(m for m in found if m not in preferred)
    return preferred + extras


def line_plot(df: pd.DataFrame, dataset: str, metric: str, ylabel: str, path: Path) -> bool:
    if metric not in df.columns:
        print(f"[SKIP] {dataset}: column '{metric}' is not present.")
        return False

    subset = df[(df["dataset"] == dataset) & (df["batch"] >= 0)].copy()
    if subset.empty:
        print(f"[SKIP] {dataset} / {metric}: no rows with batch >= 0 were found.")
        return False

    subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        print(f"[SKIP] {dataset} / {metric}: the metric has no numeric values.")
        return False

    stats = (
        subset.groupby(["method", "batch"], as_index=False)[metric]
        .agg(mean="mean", std="std")
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0

    for method in ordered_methods(stats):
        m = stats[stats["method"] == method].sort_values("batch")
        if m.empty:
            continue

        ax.plot(
            m["batch"],
            m["mean"],
            marker="o",
            markersize=3,
            label=method.replace("_", " ").title(),
        )
        plotted += 1

        if m["std"].notna().any():
            std = m["std"].fillna(0)
            ax.fill_between(
                m["batch"],
                m["mean"] - std,
                m["mean"] + std,
                alpha=0.15,
            )

    if plotted == 0:
        plt.close(fig)
        print(f"[SKIP] {dataset} / {metric}: no method produced plottable data.")
        return False

    ax.set_xlabel("Stream batch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset.upper()}: {ylabel} over the stream")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {path}")
    return True


def print_diagnostics(df: pd.DataFrame) -> None:
    print("\n=== Plot diagnostics ===")
    print("Datasets found:", sorted(df["dataset"].dropna().unique().tolist()))
    print("Methods found (original):", sorted(df["method_original"].dropna().unique().tolist()))
    print("Methods used after normalisation:", sorted(df["method"].dropna().unique().tolist()))
    batches = df["batch"].dropna()
    if not batches.empty:
        print("Batch range:", int(batches.min()), "to", int(batches.max()))
    print("Rows loaded:", len(df))
    print("========================\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/extended")
    args = parser.parse_args()

    root = Path(args.results_dir)
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(root)
    print_diagnostics(df)

    plots = [
        ("test_accuracy", "Test accuracy", "accuracy"),
        ("pseudo_precision", "Pseudo-label precision", "pseudo_precision"),
        ("pseudo_coverage", "Pseudo-label coverage", "pseudo_coverage"),
        ("parameter_change", "Relative parameter change", "parameter_change"),
        ("feature_centroid_drift", "Feature-centroid drift", "feature_drift"),
    ]

    saved = 0
    for dataset in sorted(df["dataset"].dropna().unique()):
        for metric, ylabel, suffix in plots:
            output = figdir / f"{dataset}_{suffix}.png"
            saved += int(line_plot(df, dataset, metric, ylabel, output))

    print(f"\nFinished. {saved} figure(s) saved to {figdir}")


if __name__ == "__main__":
    main()
