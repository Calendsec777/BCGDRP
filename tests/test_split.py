"""The released partitions are deterministic and cell-line-disjoint."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def test_primary_split_counts_and_disjointness():
    response = pd.read_csv(ROOT / "data" / "GDSC2_IC50.csv", index_col=0)
    cells = np.arange(len(sorted(response.index.astype(str))))
    train, remainder = train_test_split(cells, test_size=0.2, random_state=42)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=42)
    train, validation, test = map(lambda values: set(values.tolist()), (train, validation, test))
    assert (len(train), len(validation), len(test)) == (428, 54, 54)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    ordered = response.loc[sorted(response.index.astype(str))]
    assert int(ordered.iloc[sorted(test)].notna().sum().sum()) == 8_404

    frozen = json.loads((ROOT / "data" / "splits" / "primary_seed42.json").read_text())
    ids = np.asarray(sorted(response.index.astype(str)))
    assert set(frozen["train_depmap_ids"]) == set(ids[sorted(train)])
    assert set(frozen["val_depmap_ids"]) == set(ids[sorted(validation)])
    assert set(frozen["test_depmap_ids"]) == set(ids[sorted(test)])
    assert frozen["counts"] == {"train": 428, "validation": 54, "test": 54}
    assert frozen["pair_counts"]["test"] == 8_404


def test_split_checksums():
    split_dir = ROOT / "data" / "splits"
    assert {path.name for path in split_dir.glob("*.json")} == {
        "primary_seed42.json",
        "fivefold_seed42.json",
    }
    for split in split_dir.glob("*.json"):
        checksum = split.with_name(split.name + ".sha256")
        digest, relative = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)
        assert digest == hashlib.sha256(split.read_bytes()).hexdigest()
        assert relative == split.relative_to(ROOT).as_posix()
