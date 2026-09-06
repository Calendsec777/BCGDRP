#!/usr/bin/env python3
"""Create or verify the SHA-256 manifest for the complete release tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
SKIP = {MANIFEST}
SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "outputs"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}
SKIP_SUFFIXES = {".log", ".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path not in SKIP
            and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
            and path.name not in SKIP_NAMES
            and path.suffix.lower() not in SKIP_SUFFIXES
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def create() -> None:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST.name}: {len(lines)} files")


def check() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST.name}; run this script without --check first")
    recorded = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest

    current = {path.relative_to(ROOT).as_posix(): path for path in files()}
    errors = []
    for name in sorted(recorded.keys() - current.keys()):
        errors.append(f"missing: {name}")
    for name in sorted(current.keys() - recorded.keys()):
        errors.append(f"not listed: {name}")
    for name in sorted(recorded.keys() & current.keys()):
        if sha256(current[name]) != recorded[name]:
            errors.append(f"changed: {name}")
    if errors:
        raise SystemExit("manifest check failed:\n- " + "\n- ".join(errors))
    print(f"manifest: OK ({len(current)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of rewriting")
    args = parser.parse_args()
    check() if args.check else create()


if __name__ == "__main__":
    main()
