import argparse, subprocess, sys
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(); p.add_argument("--quick", action="store_true"); args = p.parse_args()
    lambdas = [10,50] if args.quick else [0,1,10,50,100,500]
    thresholds = [0.85,0.90] if args.quick else [0.80,0.85,0.90,0.95]
    seeds = [0,1] if args.quick else [0,1,2,3,4]
    records=[]
    for lam in lambdas:
        for thr in thresholds:
            out = Path(f"results/sweep/lam_{lam}_thr_{thr}")
            cmd=[sys.executable,"run_repeated.py","--methods","centroid_ewc","--seeds",*map(str,seeds),
                 "--ewc-lambda",str(lam),"--threshold",str(thr),"--output-dir",str(out)]
            subprocess.run(cmd,check=True)
            s=pd.read_csv(out/"mean_std_summary.csv").iloc[0].to_dict(); s.update({"lambda":lam,"threshold":thr}); records.append(s)
    table=pd.DataFrame(records).sort_values("accuracy_mean",ascending=False)
    Path("results/sweep").mkdir(parents=True,exist_ok=True)
    table.to_csv("results/sweep/sweep_summary.csv",index=False)
    print(table.to_string(index=False))

if __name__=="__main__": main()
