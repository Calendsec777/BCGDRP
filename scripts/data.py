#!/usr/bin/env python3
"""Validate public inputs and, when supplied locally, restricted COSMIC matrices."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--strict", action="store_true", help="Fail if restricted profiles are absent")
    args = parser.parse_args()

    response = pd.read_csv(args.data_dir / "GDSC2_IC50.csv", index_col=0)
    if response.shape != (536, 169):
        raise ValueError(f"Expected a 536 x 169 response matrix, found {response.shape}.")
    if int(response.notna().sum().sum()) != 81_467:
        raise ValueError("Expected 81,467 measured drug-cell pairs.")

    with (args.data_dir / "drug" / "morgan_encoding.pkl").open("rb") as handle:
        morgan = pickle.load(handle)
    missing_drugs = [drug for drug in response.columns if int(drug) not in morgan]
    if missing_drugs:
        raise ValueError(f"Morgan fingerprints are missing for {len(missing_drugs)} benchmark compounds.")
    invalid = [drug for drug in response.columns if len(morgan[int(drug)]) != 2048]
    if invalid:
        raise ValueError(f"Expected 2,048-bit Morgan fingerprints; {len(invalid)} have another size.")

    expected = {"geo_expression_cosmic.csv": 714, "geo_methylation_cosmic.csv": 603}
    absent = []
    for filename, width in expected.items():
        path = args.data_dir / "cell" / filename
        if not path.exists():
            absent.append(filename)
            continue
        frame = pd.read_csv(path, index_col=0)
        if frame.shape[1] != width:
            raise ValueError(f"{filename}: expected {width} features, found {frame.shape[1]}.")
        missing_cells = set(response.index.astype(str)) - set(frame.index.astype(str))
        if missing_cells:
            raise ValueError(f"{filename}: missing {len(missing_cells)} benchmark cell lines.")

    if absent and args.strict:
        raise FileNotFoundError(
            "Restricted COSMIC-derived matrices are absent: " + ", ".join(absent)
        )
    print("public inputs: OK (536 cells, 169 drugs, 81,467 pairs, 2,048-bit Morgan fingerprints)")
    if absent:
        print("restricted inputs: intentionally absent; see DATA_ACCESS.md")
    else:
        print("restricted inputs: OK (714 expression and 603 methylation features)")


if __name__ == "__main__":
    main()
