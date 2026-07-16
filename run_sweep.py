import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer settings for a quick check")
    args = parser.parse_args()
    lambdas = [10, 50] if args.quick else [0, 1, 10, 50, 100, 500]
    thresholds = [0.85, 0.90] if args.quick else [0.80, 0.85, 0.90, 0.95]
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    for lam in lambdas:
        for threshold in thresholds:
            out = f"results/sweep/lam_{lam}_thr_{threshold}"
            cmd = [sys.executable, "run_repeated.py", "--methods", "centroid_ewc",
                   "--seeds", *map(str, seeds), "--ewc-lambda", str(lam),
                   "--threshold", str(threshold), "--output-dir", out]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
