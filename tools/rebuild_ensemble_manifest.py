#!/usr/bin/env python3
"""Build a portable completion manifest from the frozen run metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUTPUT = RAW / "ensemble_manifest.csv"
FIELDS = (
    "run_id",
    "pool_seed",
    "policy",
    "status",
    "return_code",
    "wall_seconds",
    "log_path",
    "last_output",
)


def main() -> None:
    rows = []
    for path in sorted(RAW.glob("pool*_metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": metadata["run_id"],
                "pool_seed": int(metadata["pool_seed"]),
                "policy": metadata["policy"],
                "status": metadata["status"],
                "return_code": 0 if metadata["status"] == "complete" else 1,
                "wall_seconds": float(metadata["elapsed_seconds"]),
                "log_path": "",
                "last_output": "",
            }
        )
    rows.sort(key=lambda row: (row["pool_seed"], row["policy"]))
    if len(rows) != 36:
        raise SystemExit(f"Expected 36 metadata records, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
