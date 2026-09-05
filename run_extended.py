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
    p.add_argument("--resume", action="store_true", help="Skip runs whose summary JSON already exists")
    p.add_argument("--continue-on-error", action="store_true", help="Log failed runs and continue")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    seeds = [0, 1] if args.quick else args.seeds
    for dataset in args.datasets:
        for method in args.methods:
            for seed in seeds:
                if args.resume:
                    existing = list(out.glob(f"summary_{dataset}_{method}_seed{seed}_*.json"))
                    if existing:
                        print("SKIP existing:", dataset, method, "seed", seed, flush=True)
                        continue
                cmd = [sys.executable, "run_experiment.py", "--dataset", dataset,
                       "--method", method, "--seed", str(seed), "--output-dir", args.output_dir]
                if args.quick:
                    cmd += ["--stream-batches", "5", "--initial-epochs", "1", "--fisher-samples", "100"]
                print("\nRUN:", " ".join(cmd), flush=True)
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    failures.append({"dataset": dataset, "method": method, "seed": seed, "returncode": result.returncode})
                    if not args.continue_on_error:
                        raise SystemExit(result.returncode)
    if failures:
        log = out / "failed_runs.csv"
        with log.open("w") as f:
            f.write("dataset,method,seed,returncode\n")
            for item in failures:
                f.write(f"{item['dataset']},{item['method']},{item['seed']},{item['returncode']}\n")
        print("Some runs failed. See", log)


if __name__ == "__main__":
    main()
