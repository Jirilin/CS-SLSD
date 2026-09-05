from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, allow_fail: bool = False):
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Final dissertation execution pipeline")
    parser.add_argument("--quick", action="store_true", help="Run a small validation matrix first")
    parser.add_argument("--full", action="store_true", help="Run the full 3 datasets x 5 methods x 5 seeds matrix")
    parser.add_argument("--results-dir", default="results/extended")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "tests", "-v"])

    if args.quick:
        run([sys.executable, "run_extended.py", "--quick", "--output-dir", args.results_dir], args.continue_on_error)

    if args.full:
        run([sys.executable, "run_extended.py", "--output-dir", args.results_dir, "--resume", "--continue-on-error"], args.continue_on_error)

    # These steps are safe after either quick or full runs if summary/metrics exist.
    run([sys.executable, "aggregate_results.py", "--results-dir", args.results_dir], True)
    run([sys.executable, "plot_results.py", "--results-dir", args.results_dir], True)
    run([sys.executable, "run_analysis.py", "--results-dir", args.results_dir], True)
    run([sys.executable, "create_final_figures.py", "--results-dir", args.results_dir], True)
    run([sys.executable, "generate_pipeline_diagram.py"], True)
    run([sys.executable, "capture_environment.py", "--output-dir", args.results_dir], True)
    run([sys.executable, "make_submission_manifest.py", "--results-dir", args.results_dir], True)
    run([sys.executable, "validate_submission.py", "--results-dir", args.results_dir], True)

    print("\nFinal pipeline finished. Check results/extended and results/final_figures.")


if __name__ == "__main__":
    main()
