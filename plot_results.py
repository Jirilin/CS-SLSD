from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    root = Path("results/repeated")
    df = pd.read_csv(root / "all_runs.csv")
    stream = df[df.batch >= 0]
    mean_curve = stream.groupby(["method", "batch"], as_index=False).agg(
        mean_accuracy=("test_accuracy", "mean"),
        std_accuracy=("test_accuracy", "std"),
    )

    plt.figure(figsize=(9, 5))
    for method, group in mean_curve.groupby("method"):
        plt.plot(group.batch, group.mean_accuracy, label=method)
        plt.fill_between(group.batch,
                         group.mean_accuracy - group.std_accuracy.fillna(0),
                         group.mean_accuracy + group.std_accuracy.fillna(0), alpha=0.15)
    plt.xlabel("Stream batch")
    plt.ylabel("Test accuracy")
    plt.title("MNIST continual semi-supervised learning: mean ± SD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "accuracy_mean_std.png", dpi=200)
    plt.close()

    dist = pd.read_csv(next(root.glob("distribution_seed*.csv")))
    plt.figure(figsize=(9, 4))
    plt.plot(dist.batch, dist.total_variation_from_previous, marker="o")
    plt.xlabel("Stream batch")
    plt.ylabel("Total variation distance")
    plt.title("Amount of class-distribution change")
    plt.tight_layout()
    plt.savefig(root / "distribution_change.png", dpi=200)
    plt.close()

    change = stream.groupby(["method", "batch"], as_index=False).parameter_change.mean()
    plt.figure(figsize=(9, 5))
    for method, group in change.groupby("method"):
        plt.plot(group.batch, group.parameter_change, label=method)
    plt.xlabel("Stream batch")
    plt.ylabel("Relative parameter change")
    plt.title("Model parameter change after each batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "parameter_change.png", dpi=200)
    plt.close()
    print(f"Saved plots to {root}")


if __name__ == "__main__":
    main()
