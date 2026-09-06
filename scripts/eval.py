#!/usr/bin/env python3
"""Evaluate released checkpoints on the harmonized CTRPv2 pairs."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcgdrp.baseline import BANDRPGate  # noqa: E402
from bcgdrp.config import load_config  # noqa: E402
from bcgdrp.model import BCGDRP  # noqa: E402
from bcgdrp.training import _pearson, _rank  # noqa: E402


class PairDataset(Dataset):
    def __init__(self, pairs, drugs, expression, methylation, cell_index, drug_index):
        self.pairs = pairs.reset_index(drop=True)
        self.drugs = drugs
        self.expression = expression
        self.methylation = methylation
        self.cell_index = cell_index
        self.drug_index = drug_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        cell = self.cell_index[str(row.depmap_id)]
        drug = self.drug_index[str(row.gdsc_pubchem_id)]
        return self.drugs[drug], self.expression[cell], self.methylation[cell], index


def collate(batch):
    drug = [torch.stack([item[0] for item in batch])]
    expression = torch.stack([item[1] for item in batch])
    methylation = torch.stack([item[2] for item in batch])
    unused = torch.zeros((len(batch), 1), dtype=torch.float32)
    indices = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return drug, [expression, unused, methylation, unused.clone()], indices


def load_pairs(path: Path, subset: str) -> pd.DataFrame:
    pairs = pd.read_csv(path, dtype={"depmap_id": str, "gdsc_pubchem_id": str})
    required = {"depmap_id", "gdsc_pubchem_id", "resistance_rank"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Pair table is missing columns: {sorted(missing)}")
    pairs = pairs.dropna(subset=list(required)).copy()
    if subset == "no_gdsc_train":
        pairs = pairs[~pairs["in_gdsc_train"].astype(str).str.lower().eq("true")]
    elif subset == "no_gdsc_response":
        pairs = pairs[~pairs["in_gdsc_response"].astype(str).str.lower().eq("true")]
    return pairs.reset_index(drop=True)


def load_features(data_dir: Path, pairs: pd.DataFrame):
    expression_path = data_dir / "cell" / "geo_expression_cosmic.csv"
    methylation_path = data_dir / "cell" / "geo_methylation_cosmic.csv"
    missing = [path for path in (expression_path, methylation_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Restricted COSMIC matrices are required locally but are not distributed: {names}")

    expression_df = pd.read_csv(expression_path, index_col=0)
    methylation_df = pd.read_csv(methylation_path, index_col=0)
    expression_df.index = expression_df.index.astype(str)
    methylation_df.index = methylation_df.index.astype(str)
    cells = sorted(pairs.depmap_id.astype(str).unique())
    drugs = sorted(pairs.gdsc_pubchem_id.astype(str).unique(), key=int)
    unavailable_cells = sorted(set(cells) - set(expression_df.index) | set(cells) - set(methylation_df.index))
    if unavailable_cells:
        raise ValueError(f"Profiles are missing for {len(unavailable_cells)} CTRPv2 cell lines.")

    with (data_dir / "drug" / "morgan_encoding.pkl").open("rb") as handle:
        morgan = pickle.load(handle)
    unavailable_drugs = [drug for drug in drugs if int(drug) not in morgan]
    if unavailable_drugs:
        raise ValueError(f"Morgan fingerprints are missing for {len(unavailable_drugs)} compounds.")

    drug_tensor = torch.as_tensor(np.asarray([morgan[int(drug)] for drug in drugs]), dtype=torch.float32)
    expression = torch.as_tensor(expression_df.loc[cells].to_numpy(), dtype=torch.float32)
    methylation = torch.as_tensor(methylation_df.loc[cells].to_numpy(), dtype=torch.float32)
    dataset = PairDataset(
        pairs,
        drug_tensor,
        expression,
        methylation,
        {cell: index for index, cell in enumerate(cells)},
        {drug: index for index, drug in enumerate(drugs)},
    )
    return dataset, expression.shape[1], methylation.shape[1]


def load_model(model_name: str, checkpoint: Path, dimensions, device):
    cfg_name = "bcgdrp.yaml" if model_name == "bcgdrp" else "baseline.yaml"
    cfg = load_config(path=ROOT / "configs" / cfg_name)
    model_class = BCGDRP if model_name == "bcgdrp" else BANDRPGate
    model = model_class(
        cell_exp_dim=dimensions[0],
        cell_mut_dim=1,
        cell_meth_dim=dimensions[1],
        cell_path_dim=1,
        **cfg,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    indices, predictions = [], []
    for drug, cell, row_index in loader:
        drug = [value.to(device) for value in drug]
        cell = [value.to(device) for value in cell]
        output, _ = model(drug, cell)
        indices.append(row_index.numpy())
        predictions.append(output.cpu().numpy())
    return np.concatenate(indices), np.concatenate(predictions)


def summarize(frame: pd.DataFrame, model: str, seed: int, min_cells: int):
    rows = []
    for drug_id, group in frame.groupby("gdsc_pubchem_id"):
        if group.depmap_id.nunique() < min_cells:
            continue
        rho = _pearson(_rank(group.prediction.to_numpy()), _rank(group.resistance_rank.to_numpy()))
        rows.append({"model": model, "seed": seed, "gdsc_pubchem_id": drug_id, "n": len(group), "spearman": rho})
    per_drug = pd.DataFrame(rows)
    summary = {
        "model": model,
        "seed": seed,
        "n_pairs": len(frame),
        "n_drugs": len(per_drug),
        "macro_spearman": float(per_drug.spearman.mean()),
        "micro_spearman": _pearson(_rank(frame.prediction.to_numpy()), _rank(frame.resistance_rank.to_numpy())),
    }
    return per_drug, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "data" / "ctrp" / "pairs.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ctrp")
    parser.add_argument("--subset", choices=["all", "no_gdsc_train", "no_gdsc_response"], default="no_gdsc_response")
    parser.add_argument("--models", default="bcgdrp,bandrp_gate")
    parser.add_argument("--seeds", default="2020,2021,2022,2023,2024")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.pairs, args.subset)
    dataset, exp_dim, meth_dim = load_features(args.data_dir, pairs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )
    device = torch.device(args.device)
    per_drug_all, summaries = [], []
    for model_name in [value.strip() for value in args.models.split(",") if value.strip()]:
        if model_name not in {"bcgdrp", "bandrp_gate"}:
            raise ValueError(f"Unknown model: {model_name}")
        for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
            checkpoint = args.checkpoint_dir / f"{model_name}_seed{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            print(f"model={model_name} seed={seed}")
            model = load_model(model_name, checkpoint, (exp_dim, meth_dim), device)
            row_index, values = predict(model, loader, device)
            frame = pairs.copy()
            frame.loc[row_index, "prediction"] = values
            frame["model"] = model_name
            frame["seed"] = seed
            frame.to_csv(args.output_dir / f"predictions_{args.subset}_{model_name}_{seed}.csv", index=False)
            per_drug, summary = summarize(frame, model_name, seed, args.min_cells)
            per_drug_all.append(per_drug)
            summaries.append(summary)

    per_drug = pd.concat(per_drug_all, ignore_index=True)
    summary = pd.DataFrame(summaries)
    per_drug.to_csv(args.output_dir / "per_drug.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pivot = summary.pivot(index="seed", columns="model", values="macro_spearman")
    if {"bcgdrp", "bandrp_gate"}.issubset(pivot.columns):
        pivot["delta"] = pivot.bcgdrp - pivot.bandrp_gate
        pivot.to_csv(args.output_dir / "delta.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
