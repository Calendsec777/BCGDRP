#!/usr/bin/env python3
"""Regenerate 2,048-bit radius-2 Morgan fingerprints from the released SMILES."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "drug_info.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "morgan_encoding.pkl")
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    required = {"pubchem_id", "canonicalsmiles"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Drug table is missing columns: {sorted(missing)}")

    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    invalid = []
    for row in table.drop_duplicates("pubchem_id").itertuples():
        molecule = Chem.MolFromSmiles(str(row.canonicalsmiles))
        if molecule is None:
            invalid.append(int(row.pubchem_id))
            continue
        fingerprints[int(row.pubchem_id)] = generator.GetFingerprintAsNumPy(molecule)
    if invalid:
        raise ValueError(f"Invalid SMILES for PubChem IDs: {invalid}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(fingerprints, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(fingerprints)} fingerprints to {args.output}")


if __name__ == "__main__":
    main()
