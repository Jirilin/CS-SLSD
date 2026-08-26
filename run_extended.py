from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "svhn"])
    p.add_argument("--methods", nargs="+", default=["offline", "naive", "replay", "ewc", "proposed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output-dir", default="results/extended")
    p.add_argument("--quick", action="store_true", help="2 seeds, 5 batches, 1 initial epoch")
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seeds = [0, 1] if args.quick else args.seeds
    for dataset in args.datasets:
        for method in args.methods:
            for seed in seeds:
                cmd = [sys.executable, "run_experiment.py", "--dataset", dataset,
                       "--method", method, "--seed", str(seed), "--output-dir", args.output_dir]
                if args.quick:
                    cmd += ["--stream-batches", "5", "--initial-epochs", "1", "--fisher-samples", "100"]
                print("\nRUN:", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
