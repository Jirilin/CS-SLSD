from __future__ import annotations
import subprocess
import sys


def main():
    methods = ["naive", "replay", "ewc", "proposed"]
    for method in methods:
        for seed in [0, 1, 2, 3, 4]:
            subprocess.run([sys.executable, "run_experiment.py", "--dataset", "mnist",
                            "--method", method, "--seed", str(seed),
                            "--output-dir", "results/ablations"], check=True)


if __name__ == "__main__":
    main()
