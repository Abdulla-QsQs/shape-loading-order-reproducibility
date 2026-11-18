#!/usr/bin/env python3
"""Regenerate the package manifest and SHA-256 records deterministically."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_FILES = {"PACKAGE_MANIFEST.txt", "SHA256SUMS.txt"}
EXCLUDED_PARTS = {".git", "reproduction_output"}


def package_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in METADATA_FILES or any(
            part in EXCLUDED_PARTS for part in relative.parts
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = package_files()
    relative_paths = [path.relative_to(ROOT).as_posix() for path in files]
    (ROOT / "PACKAGE_MANIFEST.txt").write_text(
        "\n".join(relative_paths) + "\n", encoding="utf-8", newline="\n"
    )
    checksum_lines = [
        f"{sha256(path)}  {relative}"
        for path, relative in zip(files, relative_paths)
    ]
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"release_files={len(files)}")


if __name__ == "__main__":
    main()
