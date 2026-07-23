from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--methods", nargs="+", default=["naive","centroid","centroid_ewc","centroid_replay","centroid_replay_ewc"])
    p.add_argument("--ewc-lambda", type=float, default=50.0)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--output-dir", default="results/repeated")
    args = p.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for seed in args.seeds:
            cmd = [sys.executable, "run_experiment.py", "--method", method, "--seed", str(seed),
                   "--ewc-lambda", str(args.ewc_lambda), "--threshold", str(args.threshold),
                   "--output-dir", str(out)]
            print("Running:", " ".join(cmd), flush=True); subprocess.run(cmd, check=True)
    frames = [pd.read_csv(p) for p in out.glob("metrics_*.csv")]
    if not frames: raise RuntimeError("No metric files were generated")
    all_df = pd.concat(frames, ignore_index=True); all_df.to_csv(out / "all_runs.csv", index=False)
    stream = all_df[all_df.batch >= 0]
    per_run = stream.groupby(["method","seed"], as_index=False).agg(
        mean_accuracy=("test_accuracy","mean"), final_accuracy=("test_accuracy","last"),
        mean_precision=("pseudo_precision","mean"), mean_coverage=("pseudo_coverage","mean"),
        mean_drift=("feature_centroid_drift","mean"), mean_parameter_change=("parameter_change","mean"))
    summary = per_run.groupby("method", as_index=False).agg(
        accuracy_mean=("mean_accuracy","mean"), accuracy_std=("mean_accuracy","std"),
        final_accuracy_mean=("final_accuracy","mean"), final_accuracy_std=("final_accuracy","std"),
        precision_mean=("mean_precision","mean"), precision_std=("mean_precision","std"),
        coverage_mean=("mean_coverage","mean"), coverage_std=("mean_coverage","std"),
        centroid_drift_mean=("mean_drift","mean"), centroid_drift_std=("mean_drift","std"),
        parameter_change_mean=("mean_parameter_change","mean"), parameter_change_std=("mean_parameter_change","std"))
    per_run.to_csv(out / "per_run_summary.csv", index=False)
    summary.to_csv(out / "mean_std_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__": main()
