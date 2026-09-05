from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def safe_run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Capture environment for reproducibility")
    parser.add_argument("--output-dir", default="results/extended")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze": safe_run([sys.executable, "-m", "pip", "freeze"]),
        "git_commit": safe_run(["git", "rev-parse", "HEAD"]),
        "git_status": safe_run(["git", "status", "--short"]),
    }
    try:
        import torch
        data["torch_version"] = torch.__version__
        data["cuda_available"] = torch.cuda.is_available()
        data["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception as exc:
        data["torch_error"] = str(exc)

    (out / "environment_reproducibility.json").write_text(json.dumps(data, indent=2))
    (out / "requirements_frozen.txt").write_text(data.get("pip_freeze", ""))
    print("Saved environment files to", out)


if __name__ == "__main__":
    main()
