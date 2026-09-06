#!/usr/bin/env python3
"""Predict response for a CSV of DepMap/PubChem pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import collate, load_features, load_model, predict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, help="CSV with depmap_id and gdsc_pubchem_id columns")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=["bcgdrp", "bandrp_gate"], default="bcgdrp")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "files")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frame = pd.read_csv(args.pairs, dtype={"depmap_id": str, "gdsc_pubchem_id": str})
    missing = {"depmap_id", "gdsc_pubchem_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    frame["resistance_rank"] = 0.0
    dataset, exp_dim, meth_dim = load_features(args.data_dir, frame)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device(args.device)
    checkpoint = args.checkpoint_dir / f"{args.model}_seed{args.seed}.pt"
    model = load_model(args.model, checkpoint, (exp_dim, meth_dim), device)
    row_index, values = predict(model, loader, device)
    frame.loc[row_index, "pred_logIC50"] = values
    frame.drop(columns="resistance_rank").to_csv(args.output, index=False)
    print(f"wrote {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
