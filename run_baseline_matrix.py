from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["mnist", "cifar10", "svhn"]
DEFAULT_METHODS = ["offline", "naive", "replay", "ewc"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def already_done(output_dir: Path, dataset: str, method: str, seed: int) -> bool:
    pattern = f"summary_{dataset}_{method}_seed{seed}_*.json"
    return any(output_dir.glob(pattern))


def main():
    p = argparse.ArgumentParser(
        description="Run the complete repeated baseline experiment matrix."
    )
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Run even when a matching summary JSON already exists.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later runs if one run fails.",
    )
    args = p.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.methods) * len(args.seeds)
    number = 0
    failures = []

    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                number += 1

                if not args.rerun and already_done(output, dataset, method, seed):
                    print(
                        f"[{number}/{total}] SKIP "
                        f"{dataset} / {method} / seed={seed} (already complete)"
                    )
                    continue

                print("\n" + "=" * 88)
                print(
                    f"[{number}/{total}] RUN "
                    f"dataset={dataset} method={method} seed={seed}"
                )
                print("=" * 88)

                cmd = [
                    sys.executable,
                    "run_experiment.py",
                    "--dataset", dataset,
                    "--method", method,
                    "--seed", str(seed),
                    "--output-dir", str(output),
                ]
                if args.smoke:
                    cmd.append("--smoke")

                result = subprocess.run(cmd)

                if result.returncode != 0:
                    failures.append((dataset, method, seed, result.returncode))
                    print(
                        f"FAILED: dataset={dataset}, method={method}, "
                        f"seed={seed}, exit={result.returncode}"
                    )
                    if not args.continue_on_error:
                        raise SystemExit(result.returncode)

    print("\nBaseline matrix finished.")
    if failures:
        print("\nFailures:")
        for item in failures:
            print("  ", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
