"""Run the exact CRIT v0.4 transfer-identifiability experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import scipy

from crit_v04.synthetic import run_identifiability_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args()
    result = {
        "benchmark": "CRIT v0.4 utility-sensitive transfer identifiability",
        "date": date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "result": run_identifiability_benchmark(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
