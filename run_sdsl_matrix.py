from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "svhn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output-dir", default="results/sdsl_full")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    total = len(args.datasets) * len(args.seeds)
    done = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            done += 1
            cmd = [sys.executable, "run_sdsl_full.py", "--dataset", dataset,
                   "--seed", str(seed), "--output-dir", args.output_dir]
            if args.quick:
                cmd += ["--stream-batches", "5", "--initial-epochs", "1", "--replay-epochs", "1"]
            print(f"\n=== SDSL run {done}/{total}: {dataset}, seed={seed} ===")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
