from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/sdsl_full")
    args = p.parse_args()
    root = Path(args.results_dir)
    records = []
    for path in sorted(root.glob("summary_*_sdsl_full_seed*.json")):
        with open(path) as f:
            d = json.load(f)
        d.pop("config", None)
        records.append(d)
    if not records:
        raise SystemExit(f"No SDSL summary JSON files found in {root}")
    df = pd.DataFrame(records)
    df.to_csv(root / "sdsl_runs.csv", index=False)
    numeric = [c for c in df.columns if c not in {"dataset", "method", "seed"}]
    rows = []
    for dataset, g in df.groupby("dataset"):
        row = {"dataset": dataset, "runs": len(g)}
        for c in numeric:
            if pd.api.types.is_numeric_dtype(g[c]):
                row[c + "_mean"] = g[c].mean()
                row[c + "_std"] = g[c].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(root / "sdsl_mean_std.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
