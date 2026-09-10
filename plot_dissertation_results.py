from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
SDSL_DIR = RESULTS_DIR / "sdsl_full"
FIGURE_DIR = RESULTS_DIR / "figures"
AGG_DIR = RESULTS_DIR / "aggregated"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
AGG_DIR.mkdir(parents=True, exist_ok=True)



def save_figure(filename):
    path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    return None



def plot_stream_accuracy(dataset):
    files = sorted(
        SDSL_DIR.glob(
            f"metrics_{dataset}_sdsl_full_seed*.csv"
        )
    )

    if not files:
        print(
            f"No stream metric files found for {dataset}"
        )
        return

    runs = []

    for file in files:
        df = pd.read_csv(file)

        batch_col = find_column(
            df,
            [
                "batch",
                "stream_batch",
                "batch_id",
                "step"
            ]
        )

        accuracy_col = find_column(
            df,
            [
                "accuracy",
                "test_accuracy",
                "overall_accuracy",
                "acc"
            ]
        )

        if batch_col is None or accuracy_col is None:
            print(
                f"Skipping {file.name}: "
                "batch/accuracy column not found"
            )
            continue

        temp = df[
            [batch_col, accuracy_col]
        ].copy()

        temp.columns = [
            "batch",
            "accuracy"
        ]

        temp["seed"] = file.stem.split("seed")[-1]

        runs.append(temp)

    if not runs:
        return

    data = pd.concat(
        runs,
        ignore_index=True
    )

    summary = (
        data
        .groupby("batch")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0)

    x = summary["batch"].to_numpy()
    mean = summary["mean"].to_numpy()
    std = summary["std"].to_numpy()

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.20
    )

    plt.xlabel("Stream batch")
    plt.ylabel("Test accuracy")
    plt.title(
        f"{dataset.upper()} accuracy across the data stream"
    )

    plt.ylim(0, 1)
    plt.grid(alpha=0.25)

    save_figure(
        f"accuracy_stream_{dataset}.png"
    )


def load_sdsl_summaries():
    rows = []

    for file in sorted(
        SDSL_DIR.glob(
            "summary_*_sdsl_full_seed*.json"
        )
    ):
        with open(file, "r") as f:
            result = json.load(f)

        rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    output = AGG_DIR / "sdsl_runs.csv"
    df.to_csv(output, index=False)

    print(f"Saved: {output}")

    return df


def plot_final_accuracy(sdsl):
    if sdsl.empty:
        return

    summary = (
        sdsl
        .groupby("dataset")["final_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        summary["dataset"],
        summary["mean"],
        yerr=summary["std"],
        capsize=5
    )

    plt.ylabel("Final accuracy")
    plt.xlabel("Dataset")
    plt.title(
        "Final SDSL accuracy across datasets"
    )

    plt.ylim(0, 1)

    save_figure(
        "final_accuracy_sdsl.png"
    )


def plot_forgetting(sdsl):
    if sdsl.empty:
        return

    summary = (
        sdsl
        .groupby("dataset")[
            "classwise_forgetting"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        summary["dataset"],
        summary["mean"],
        yerr=summary["std"],
        capsize=5
    )

    plt.ylabel("Class-wise forgetting")
    plt.xlabel("Dataset")
    plt.title(
        "Catastrophic forgetting across datasets"
    )

    save_figure(
        "forgetting_sdsl.png"
    )



def plot_pseudo_quality(sdsl):
    if sdsl.empty:
        return

    columns = [
        "mean_initial_pseudo_precision",
        "mean_cluster_precision",
        "mean_semantic_precision",
        "mean_pseudo_precision"
    ]

    labels = [
        "Initial",
        "Clustering",
        "Semantic",
        "Final"
    ]

    grouped = (
        sdsl
        .groupby("dataset")[columns]
        .mean()
    )

    for dataset in grouped.index:

        values = grouped.loc[
            dataset,
            columns
        ].to_numpy()

        plt.figure(figsize=(8, 5))

        plt.bar(
            labels,
            values
        )

        plt.ylabel("Pseudo-label precision")
        plt.xlabel("Pseudo-label stage")

        plt.title(
            f"{dataset.upper()} pseudo-label quality"
        )

        plt.ylim(0, 1)

        save_figure(
            f"pseudo_quality_{dataset}.png"
        )


def plot_flatness(sdsl):
    if sdsl.empty:
        return

    grouped = (
        sdsl
        .groupby("dataset")
        .agg(
            sharpness=(
                "mean_sharpness_gap",
                "mean"
            ),
            width=(
                "mean_flat_width",
                "mean"
            )
        )
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    plt.scatter(
        grouped["width"],
        grouped["sharpness"],
        s=100
    )

    for _, row in grouped.iterrows():
        plt.annotate(
            row["dataset"].upper(),
            (
                row["width"],
                row["sharpness"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Mean perturbation width")
    plt.ylabel("Mean sharpness gap")

    plt.title(
        "Flat-region optimisation diagnostics"
    )

    plt.grid(alpha=0.25)

    save_figure(
        "flatness_diagnostics.png"
    )



def plot_runtime(sdsl):
    if sdsl.empty:
        return

    summary = (
        sdsl
        .groupby("dataset")[
            "mean_batch_seconds"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        summary["dataset"],
        summary["mean"],
        yerr=summary["std"],
        capsize=5
    )

    plt.xlabel("Dataset")
    plt.ylabel("Seconds per stream batch")

    plt.title(
        "Computational cost of SDSL"
    )

    save_figure(
        "runtime_sdsl.png"
    )


def plot_method_comparison():

    file = (
        RESULTS_DIR /
        "final_method_comparison.csv"
    )

    if not file.exists():
        print(
            "Method comparison CSV not found yet."
        )
        return

    df = pd.read_csv(file)

    required = {
        "dataset",
        "method",
        "final_accuracy"
    }

    if not required.issubset(df.columns):
        print(
            "Comparison CSV does not contain "
            "expected columns."
        )
        return

    summary = (
        df
        .groupby(
            ["dataset", "method"]
        )["final_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    for dataset in summary["dataset"].unique():

        temp = summary[
            summary["dataset"] == dataset
        ]

        plt.figure(figsize=(9, 5))

        plt.bar(
            temp["method"],
            temp["mean"],
            yerr=temp["std"],
            capsize=5
        )

        plt.ylabel("Final accuracy")
        plt.xlabel("Continual learning method")

        plt.title(
            f"{dataset.upper()} method comparison"
        )

        plt.ylim(0, 1)
        plt.xticks(rotation=20)

        save_figure(
            f"method_comparison_{dataset}.png"
        )


def main():

    print(
        "\nGenerating dissertation figures...\n"
    )

    for dataset in [
        "mnist",
        "cifar10",
        "svhn"
    ]:
        plot_stream_accuracy(dataset)

    sdsl = load_sdsl_summaries()

    plot_final_accuracy(sdsl)
    plot_forgetting(sdsl)
    plot_pseudo_quality(sdsl)
    plot_flatness(sdsl)
    plot_runtime(sdsl)
    plot_method_comparison()

    print(
        "\nFinished generating figures.\n"
    )


if __name__ == "__main__":
    main()