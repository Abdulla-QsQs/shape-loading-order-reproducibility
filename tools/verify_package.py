#!/usr/bin/env python3
"""Verify integrity, provenance records, portability, and release completeness."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_FILES = {"PACKAGE_MANIFEST.txt", "SHA256SUMS.txt"}
EXCLUDED_PARTS = {".git", "reproduction_output"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitattributes",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".xdmf",
    ".xml",
    ".yaml",
    ".yml",
}
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/Users/", re.IGNORECASE),
    re.compile(chr(47) + r"home/[^/\s]+(?:/|$)", re.IGNORECASE),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_paths() -> list[str]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in METADATA_FILES or any(
            part in EXCLUDED_PARTS for part in relative.parts
        ):
            continue
        paths.append(relative.as_posix())
    return sorted(paths)


def verify_manifest_and_checksums() -> int:
    manifest = [
        line
        for line in (ROOT / "PACKAGE_MANIFEST.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    actual = package_paths()
    require(manifest == actual, "PACKAGE_MANIFEST.txt does not match the release files")

    checksum_rows = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_rows[relative] = digest
    require(
        sorted(checksum_rows) == manifest,
        "SHA256SUMS.txt file set does not match the manifest",
    )
    for relative in manifest:
        require(
            sha256(ROOT / relative) == checksum_rows[relative],
            f"Checksum mismatch: {relative}",
        )
    return len(manifest)


def verify_no_private_paths(paths: list[str]) -> None:
    leaks = []
    for relative in paths:
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            ".gitattributes",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS):
            leaks.append(relative)
    require(not leaks, f"Private absolute paths remain: {leaks}")


def verify_pools() -> None:
    manifest_path = ROOT / "inputs" / "pool_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 12, "Direction-pool manifest must contain 12 pools")
    require(
        {int(row["pool_seed"]) for row in rows} == set(range(4101, 4113)),
        "Pool seeds are incomplete",
    )
    for row in rows:
        path = ROOT / "inputs" / "direction_pools" / row["source_file"]
        require(path.exists(), f"Missing direction pool: {path.name}")
        require(
            sha256(path) == row["sha256"],
            f"Direction-pool hash mismatch: {path.name}",
        )


def verify_run_records() -> None:
    metadata_paths = sorted((ROOT / "results" / "raw").glob("pool*_metadata.json"))
    require(len(metadata_paths) == 36, "Expected 36 run metadata records")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    require(
        all(row["status"] == "complete" for row in records),
        "A publication run is not complete",
    )
    require(
        all(int(row["completed_cycles"]) == 12 for row in records),
        "A publication run has fewer than 12 cycles",
    )
    require(
        Counter(row["policy"] for row in records)
        == {"fixed": 12, "random": 12, "feedback": 12},
        "Policy counts are not 12/12/12",
    )
    require(
        len({row["configuration_sha256"] for row in records}) == 1,
        "Run configuration hashes differ",
    )

    with (
        ROOT / "supplementary_material" / "S3_publication_run_beta_audit.csv"
    ).open(newline="", encoding="utf-8") as handle:
        beta_rows = list(csv.DictReader(handle))
    require(len(beta_rows) == 36, "Beta audit must contain 36 rows")
    require(
        {float(row["feedback_strength"]) for row in beta_rows} == {250.0},
        "Publication beta is not uniformly 250",
    )

    with (
        ROOT
        / "supplementary_material"
        / "S4_beta250_selector_diagnostics_all_steps.csv"
    ).open(newline="", encoding="utf-8") as handle:
        selector_rows = list(csv.DictReader(handle))
    require(
        len(selector_rows) == 132,
        "Expected 132 nonterminal feedback selections",
    )
    entropies = [float(row["selection_entropy"]) for row in selector_rows]
    require(
        min(entropies) >= 0.0 and max(entropies) <= 1.0,
        "Selection entropy is outside [0, 1]",
    )


def verify_release_record() -> None:
    version = (ROOT / "VERSION.txt").read_text(encoding="utf-8")
    require(
        "package_version=reproducibility-package-v1.0.3" in version,
        "Package version is incorrect",
    )
    require(
        "release_status=release_candidate" in version,
        "Package must identify itself as a release candidate",
    )
    require("planned_tag=v1.0.3" in version, "Planned release tag is missing")
    require(
        "repository_url=https://github.com/Abdulla-QsQs/shape-loading-order-reproducibility"
        in version,
        "Repository URL is missing",
    )
    require((ROOT / "LICENSE.md").exists(), "Licence mapping is missing")
    require((ROOT / "LICENSES" / "MIT.txt").exists(), "MIT licence text is missing")
    require(
        (ROOT / "LICENSES" / "CC-BY-4.0.txt").exists(),
        "CC BY 4.0 notice is missing",
    )
    require(not list(ROOT.rglob("*.pyc")), "Compiled Python cache files remain")


def main() -> None:
    count = verify_manifest_and_checksums()
    paths = package_paths()
    verify_no_private_paths(paths)
    verify_pools()
    verify_run_records()
    verify_release_record()
    require(
        (ROOT / "results" / "raw" / "pool4101_feedback_fields.xdmf").exists(),
        "Representative XDMF is missing",
    )
    require(
        (ROOT / "results" / "raw" / "pool4101_feedback_fields.h5").exists(),
        "Representative HDF5 is missing",
    )
    print(
        f"package_verified=true files={count} runs=36 pools=12 selector_steps=132"
    )


if __name__ == "__main__":
    main()
