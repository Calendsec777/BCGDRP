"""Integrity tests for redistributable model inputs."""

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_response_matrix():
    response = pd.read_csv(ROOT / "data" / "GDSC2_IC50.csv", index_col=0)
    assert response.shape == (536, 169)
    assert int(response.notna().sum().sum()) == 81_467
    assert response.index.is_unique
    assert response.columns.is_unique


def test_morgan_coverage():
    response = pd.read_csv(ROOT / "data" / "GDSC2_IC50.csv", index_col=0)
    with (ROOT / "data" / "drug" / "morgan_encoding.pkl").open("rb") as handle:
        fingerprints = pickle.load(handle)
    assert all(int(identifier) in fingerprints for identifier in response.columns)
    assert all(len(fingerprints[int(identifier)]) == 2048 for identifier in response.columns)


def test_restricted_files_are_not_distributed():
    names = {
        "geo_expression_cosmic.csv",
        "geo_methylation_cosmic.csv",
        "geo_mutation_cosmic.csv",
        "pathway_cosmic.csv",
    }
    assert not [path for path in ROOT.rglob("*.csv") if path.name in names]
