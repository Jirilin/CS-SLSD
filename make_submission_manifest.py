from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pandas as pd

EXCLUDE_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", "data"}
EXCLUDE_EXTENSIONS = {".pyc"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_EXTENSIONS:
        return False
    return path.is_file()


def main():
    parser = argparse.ArgumentParser(description="Create a manifest of files to submit or archive")
    parser.add_argument("--results-dir", default="results/extended")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(root.rglob("*")):
        if include(path):
            rel = path.relative_to(root).as_posix()
            rows.append({
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "submission_manifest.csv", index=False)
    print("Saved", out_dir / "submission_manifest.csv")


if __name__ == "__main__":
    main()
