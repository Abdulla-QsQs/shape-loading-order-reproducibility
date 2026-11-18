#!/usr/bin/env python3
"""Require regenerated processed outputs to match the frozen publication files."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def normalized_text_sha256(path: Path) -> str:
    """Hash text content while treating LF and CRLF as equivalent."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: compare_processed_outputs.py REFERENCE GENERATED")
    reference = Path(sys.argv[1]).resolve()
    generated = Path(sys.argv[2]).resolve()
    expected = sorted(path.name for path in reference.iterdir() if path.is_file())
    observed = sorted(path.name for path in generated.iterdir() if path.is_file())
    if observed != expected:
        raise SystemExit(
            f"Processed file-set mismatch: expected={expected}, observed={observed}"
        )
    mismatches = [
        name
        for name in expected
        if normalized_text_sha256(reference / name)
        != normalized_text_sha256(generated / name)
    ]
    if mismatches:
        raise SystemExit(f"Processed output mismatch: {mismatches}")
    print(f"processed_outputs_match={len(expected)} (line endings normalized)")


if __name__ == "__main__":
    main()
