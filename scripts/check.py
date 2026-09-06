#!/usr/bin/env python3
"""Fail if the release tree contains restricted data, local paths, or build debris."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = {
    "geo_expression_cosmic.csv",
    "geo_methylation_cosmic.csv",
    "geo_mutation_cosmic.csv",
    "pathway_cosmic.csv",
}
DEBRIS = {"__pycache__", ".pytest_cache", ".DS_Store", "Thumbs.db"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".tex", ".cff", ".txt"}
LOCAL_PATTERNS = [r"/home/gaoyukun", r"[A-Za-z]:\\Users\\", r"SUBMISSION_NatureCommunications_BCGDRP"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors = []
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.name in DEBRIS:
            errors.append(f"build debris: {path.relative_to(ROOT)}")
        if path.is_file() and path.name in RESTRICTED:
            errors.append(f"restricted matrix: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in {".pyc", ".log"}:
            errors.append(f"generated file: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in LOCAL_PATTERNS:
                if re.search(pattern, text):
                    errors.append(f"local path marker in {path.relative_to(ROOT)}: {pattern}")

    manifest = ROOT / "checkpoints" / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            checkpoint = ROOT / "checkpoints" / "files" / f"{row['model']}_seed{row['seed']}.pt"
            if not checkpoint.exists():
                errors.append(f"missing checkpoint: {checkpoint.relative_to(ROOT)}")
            elif checkpoint.stat().st_size != int(row["bytes"]):
                errors.append(f"checkpoint size mismatch: {checkpoint.name}")
            elif sha256(checkpoint) != row["sha256"]:
                errors.append(f"checkpoint checksum mismatch: {checkpoint.name}")

    split_dir = ROOT / "data" / "splits"
    expected_splits = {"primary_seed42.json", "fivefold_seed42.json"}
    present_splits = {path.name for path in split_dir.glob("*.json")}
    if present_splits != expected_splits:
        errors.append(
            "split files differ from expected set: "
            f"expected {sorted(expected_splits)}, found {sorted(present_splits)}"
        )
    for split in sorted(split_dir.glob("*.json")):
        checksum = split.with_name(split.name + ".sha256")
        if not checksum.exists():
            errors.append(f"missing split checksum: {checksum.relative_to(ROOT)}")
            continue
        parts = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
        expected_name = split.relative_to(ROOT).as_posix()
        if len(parts) != 2:
            errors.append(f"malformed split checksum: {checksum.relative_to(ROOT)}")
        elif parts[0].lower() != sha256(split):
            errors.append(f"split checksum mismatch: {split.name}")
        elif parts[1] != expected_name:
            errors.append(f"split checksum path mismatch: {split.name}")

    if errors:
        raise SystemExit("release check failed:\n- " + "\n- ".join(errors))
    print("release tree: OK")


if __name__ == "__main__":
    main()
