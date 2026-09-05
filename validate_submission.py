from pathlib import Path
import argparse
import csv

REQUIRED_CODE = [
    "config.py", "models.py", "dataset_stream.py", "centroid_pseudolabel.py",
    "ewc.py", "replay_buffer.py", "metrics.py", "run_experiment.py",
    "run_extended.py", "aggregate_results.py", "plot_results.py",
    "run_analysis.py", "run_ablations.py",
]
REQUIRED_DOCS = [
    "README.md",
    "report_drafts/REVISED_ANALYSIS_DISCUSSION.md",
    "report_drafts/CONCLUSION_FUTURE_WORK.md",
    "report_drafts/FRONT_MATTER_TEMPLATE.md",
    "docs/PAPERWORK_ETHICS_CHECKLIST.md",
    "docs/FINAL_PRESENTATION_STRUCTURE.md",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/extended")
    args = p.parse_args()
    checks = []

    for rel in REQUIRED_CODE:
        checks.append(("code", rel, Path(rel).exists()))
    for rel in REQUIRED_DOCS:
        checks.append(("documentation", rel, Path(rel).exists()))

    results = Path(args.results_dir)
    result_expectations = [
        results / "comparison_mean_std.csv",
        results / "dissertation_comparison_table.csv",
    ]
    for path in result_expectations:
        checks.append(("results", str(path), path.exists()))

    # At least one summary for every dataset/method is expected for final evidence.
    for dataset in ["mnist", "cifar10", "svhn"]:
        for method in ["offline", "naive", "replay", "ewc", "proposed"]:
            matches = list(results.glob(f"summary_{dataset}_{method}_seed*.json"))
            checks.append(("experiment", f"{dataset}:{method} summaries", len(matches) >= 5))

    out = Path("results/submission_readiness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "item", "ready"])
        writer.writerows(checks)

    passed = sum(ok for _, _, ok in checks)
    print(f"Readiness checks: {passed}/{len(checks)} passed")
    for category, item, ok in checks:
        print(f"[{'OK' if ok else 'MISSING'}] {category}: {item}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
