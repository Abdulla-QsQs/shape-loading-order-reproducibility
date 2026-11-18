#!/usr/bin/env python3
"""Run paired matched-pool policies as isolated FEniCS subprocesses."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUN = ROOT / "src" / "matched_pool_study.py"


@dataclass(frozen=True)
class RunSpec:
    pool_seed: int
    policy: str
    policy_seed: int
    run_id: str
    write_xdmf: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--policies", nargs="+", choices=("fixed", "random", "feedback"), required=True
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--events-override", type=int)
    parser.add_argument("--mesh-resolution-override", type=int)
    parser.add_argument("--peak-force-override", type=float)
    parser.add_argument("--feedback-strength-override", type=float)
    parser.add_argument(
        "--representative-xdmf",
        action="store_true",
        help="Write XDMF fields for the first pool's feedback run.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed members.")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> List[RunSpec]:
    specs = []
    first_pool = int(args.pool_seeds[0])
    policy_offsets = {"fixed": 11, "random": 23, "feedback": 37}
    for pool_seed in args.pool_seeds:
        for policy in args.policies:
            run_id = f"pool{int(pool_seed)}_{policy}"
            specs.append(
                RunSpec(
                    pool_seed=int(pool_seed),
                    policy=policy,
                    policy_seed=int(pool_seed) * 1000 + policy_offsets[policy],
                    run_id=run_id,
                    write_xdmf=bool(
                        args.representative_xdmf
                        and int(pool_seed) == first_pool
                        and policy == "feedback"
                    ),
                )
            )
    return specs


def expected_configuration(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    if args.events_override is not None:
        config["study"]["events_per_pool"] = int(args.events_override)
    if args.mesh_resolution_override is not None:
        config["geometry"]["mesh_resolution"] = int(args.mesh_resolution_override)
    if args.peak_force_override is not None:
        config["study"]["peak_force"] = float(args.peak_force_override)
    if args.feedback_strength_override is not None:
        config["selection"]["feedback_strength"] = float(
            args.feedback_strength_override
        )
    return config


def configuration_hash(config: dict) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_complete(
    output_dir: Path,
    spec: RunSpec,
    expected_config_hash: str,
    expected_cycles: int,
) -> bool:
    metadata_path = output_dir / f"{spec.run_id}_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        metadata.get("status") == "complete"
        and int(metadata.get("pool_seed", -1)) == spec.pool_seed
        and metadata.get("policy") == spec.policy
        and int(metadata.get("policy_seed", -1)) == spec.policy_seed
        and metadata.get("configuration_sha256") == expected_config_hash
        and int(metadata.get("requested_cycles", -1)) == expected_cycles
    )


def execute(
    spec: RunSpec,
    args: argparse.Namespace,
    output_dir: Path,
    expected_config_hash: str,
    expected_cycles: int,
) -> dict:
    start = time.perf_counter()
    if not args.force and is_complete(
        output_dir, spec, expected_config_hash, expected_cycles
    ):
        return {
            "run_id": spec.run_id,
            "pool_seed": spec.pool_seed,
            "policy": spec.policy,
            "status": "skipped_complete",
            "return_code": 0,
            "wall_seconds": 0.0,
        }

    command = [
        sys.executable,
        str(SINGLE_RUN),
        "--config",
        str(Path(args.config).resolve()),
        "--pool-seed",
        str(spec.pool_seed),
        "--policy",
        spec.policy,
        "--policy-seed",
        str(spec.policy_seed),
        "--run-id",
        spec.run_id,
        "--output-dir",
        str(output_dir),
    ]
    if spec.write_xdmf:
        command.append("--write-xdmf")
    for flag, value in (
        ("--events-override", args.events_override),
        ("--mesh-resolution-override", args.mesh_resolution_override),
        ("--peak-force-override", args.peak_force_override),
        ("--feedback-strength-override", args.feedback_strength_override),
    ):
        if value is not None:
            command.extend([flag, str(value)])

    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = output_dir / f"{spec.run_id}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "run_id": spec.run_id,
        "pool_seed": spec.pool_seed,
        "policy": spec.policy,
        "status": "complete" if completed.returncode == 0 else "failed",
        "return_code": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - start),
        "log_path": log_path.name,
        "last_output": completed.stdout[-1000:],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_config = expected_configuration(args)
    expected_config_hash = configuration_hash(expected_config)
    expected_cycles = int(expected_config["study"]["events_per_pool"])
    specs = build_specs(args)
    records = []
    manifest_path = output_dir / "ensemble_manifest.csv"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {
            executor.submit(
                execute,
                spec,
                args,
                output_dir,
                expected_config_hash,
                expected_cycles,
            ): spec
            for spec in specs
        }
        for future in concurrent.futures.as_completed(future_map):
            record = future.result()
            records.append(record)
            pd.DataFrame(records).sort_values(["pool_seed", "policy"]).to_csv(
                manifest_path, index=False
            )
            print(
                f"{record['run_id']}: {record['status']} "
                f"({record['wall_seconds']:.1f}s)",
                flush=True,
            )
            if record["status"] == "failed":
                print(record.get("last_output", ""), flush=True)

    failures = [record for record in records if record["status"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
