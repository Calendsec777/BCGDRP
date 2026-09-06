"""Data loading and the deterministic cell-line-disjoint split."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class DrugResponseDataset(Dataset):
    def __init__(self, drug, expression, methylation, pairs, positions):
        self.drug = drug
        self.expression = expression
        self.methylation = methylation
        self.pairs = pairs
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        cell_idx, drug_idx, label = self.pairs[self.positions[item]]
        return self.drug[drug_idx], self.expression[cell_idx], self.methylation[cell_idx], label, cell_idx


def collate_batch(batch):
    drug = [torch.stack([item[0] for item in batch], 0)]
    expression = torch.stack([item[1] for item in batch], 0)
    methylation = torch.stack([item[2] for item in batch], 0)
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    cell_idx = torch.tensor([item[4] for item in batch], dtype=torch.long)

    # The released architectures use expression and methylation only. The two
    # one-column placeholders preserve the historical call signature without
    # requiring unused mutation/pathway matrices.
    unused = torch.zeros((len(batch), 1), dtype=torch.float32)
    return [drug, expression, unused, methylation, unused.clone(), labels, cell_idx]


def load_data(**cfg):
    """Load the three model inputs used in the final manuscript."""
    paths = cfg["path"]
    response = pd.read_csv(paths["response"], index_col=0)
    expression = pd.read_csv(paths["expression"], index_col=0)
    methylation = pd.read_csv(paths["methylation"], index_col=0)

    response.index = response.index.astype(str)
    expression.index = expression.index.astype(str)
    methylation.index = methylation.index.astype(str)
    missing = sorted(set(response.index) - set(expression.index) | set(response.index) - set(methylation.index))
    if missing:
        raise ValueError(f"Molecular profiles are missing for {len(missing)} response-matrix cell lines.")

    with Path(paths["morgan"]).open("rb") as handle:
        morgan = pickle.load(handle)

    pairs = []
    for cell_id, row in response.iterrows():
        for drug_id in response.columns:
            if not pd.isna(row[drug_id]):
                pairs.append([cell_id, str(drug_id), float(row[drug_id])])

    drug_features = {}
    for drug_id in response.columns.astype(str):
        key = int(drug_id)
        if key not in morgan:
            raise KeyError(f"Morgan fingerprint missing for PubChem ID {drug_id}")
        drug_features[drug_id] = [morgan[key]]

    return (
        drug_features,
        expression,
        methylation,
        pairs,
        response.index.to_numpy(copy=True),
        response.columns.astype(str).to_numpy(copy=True),
    )


def make_loaders(
    drug_features,
    expression,
    methylation,
    pairs,
    cell_ids,
    drug_ids,
    *,
    batch_size=128,
    num_workers=0,
    train_shuffle=True,
):
    """Return train, test and validation loaders for the frozen seed-42 split."""
    cell_ids = np.sort(np.asarray(cell_ids, dtype=str))
    drug_ids = np.sort(np.asarray(drug_ids, dtype=str))
    cell_map = {name: idx for idx, name in enumerate(cell_ids)}
    drug_map = {name: idx for idx, name in enumerate(drug_ids)}
    indexed_pairs = [[cell_map[c], drug_map[str(d)], y] for c, d, y in pairs]

    drug = []
    for drug_id in drug_ids:
        drug.append(torch.as_tensor(drug_features[str(drug_id)][0], dtype=torch.float32))
    drug = torch.stack(drug, 0)
    expression_t = torch.as_tensor(expression.loc[cell_ids].to_numpy(), dtype=torch.float32)
    methylation_t = torch.as_tensor(methylation.loc[cell_ids].to_numpy(), dtype=torch.float32)

    unique_cells = np.array(sorted({pair[0] for pair in indexed_pairs}))
    train_cells, remainder = train_test_split(unique_cells, test_size=0.2, random_state=42)
    val_cells, test_cells = train_test_split(remainder, test_size=0.5, random_state=42)
    train_cells, val_cells, test_cells = map(lambda x: set(x.tolist()), (train_cells, val_cells, test_cells))
    assert train_cells.isdisjoint(val_cells)
    assert train_cells.isdisjoint(test_cells)
    assert val_cells.isdisjoint(test_cells)

    positions = {
        "train": [i for i, pair in enumerate(indexed_pairs) if pair[0] in train_cells],
        "test": [i for i, pair in enumerate(indexed_pairs) if pair[0] in test_cells],
        "val": [i for i, pair in enumerate(indexed_pairs) if pair[0] in val_cells],
    }

    def loader(name, shuffle):
        dataset = DrugResponseDataset(drug, expression_t, methylation_t, indexed_pairs, positions[name])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_batch,
        )

    return loader("train", train_shuffle), loader("test", False), loader("val", False)


# Concise compatibility names used by the released training scripts.
dataload = load_data
data_process = make_loaders
