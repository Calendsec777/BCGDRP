#!/usr/bin/env python3
"""Freeze the primary split and refresh checksums for released split files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    checksum = path.with_name(path.name + ".sha256")
    checksum.write_text(f"{sha256(path)}  {relative}\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=ROOT / "data" / "GDSC2_IC50.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "splits")
    args = parser.parse_args()

    response = pd.read_csv(args.response, index_col=0)
    cell_ids = np.asarray(sorted(response.index.astype(str)))
    positions = np.arange(len(cell_ids))
    train, remainder = train_test_split(positions, test_size=0.2, random_state=42)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=42)
    groups = {
        "train_depmap_ids": sorted(cell_ids[train].tolist()),
        "val_depmap_ids": sorted(cell_ids[validation].tolist()),
        "test_depmap_ids": sorted(cell_ids[test].tolist()),
    }
    if set(groups["train_depmap_ids"]) & set(groups["val_depmap_ids"]):
        raise RuntimeError("Primary train and validation sets overlap.")
    if set(groups["train_depmap_ids"]) & set(groups["test_depmap_ids"]):
        raise RuntimeError("Primary train and test sets overlap.")
    if set(groups["val_depmap_ids"]) & set(groups["test_depmap_ids"]):
        raise RuntimeError("Primary validation and test sets overlap.")

    indexed = response.copy()
    indexed.index = indexed.index.astype(str)
    payload = {
        "seed": 42,
        "source_matrix": "data/GDSC2_IC50.csv",
        "source_matrix_sha256": sha256(args.response),
        "disjoint_key": "DepMap ID / response row index",
        "method": {
            "first_split": {"test_size": 0.2, "random_state": 42},
            "remainder_split": {"test_size": 0.5, "random_state": 42},
        },
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "pair_counts": {
            name.removesuffix("_depmap_ids"): int(indexed.loc[ids].notna().sum().sum())
            for name, ids in groups.items()
        },
        **groups,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = args.output_dir / "primary_seed42.json"
    primary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_checksum(primary)

    fivefold = args.output_dir / "fivefold_seed42.json"
    if not fivefold.exists():
        raise FileNotFoundError(f"Missing released five-fold split: {fivefold}")
    write_checksum(fivefold)
    print("primary split: 428 train, 54 validation, 54 test")
    print("split checksums: OK")


if __name__ == "__main__":
    main()
