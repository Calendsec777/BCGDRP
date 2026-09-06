#!/usr/bin/env python3
"""Train the BANDRP-Gate baseline on the manuscript split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcgdrp.baseline import BANDRPGate  # noqa: E402
from bcgdrp.config import configure_paths, load_config  # noqa: E402
from bcgdrp.data import load_data, make_loaders  # noqa: E402
from bcgdrp.training import fit  # noqa: E402
from bcgdrp.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "bandrp_gate")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cfg = configure_paths(load_config(path=args.config), args.data_dir, args.output_dir)
    drug, expression, methylation, pairs, cell_ids, drug_ids = load_data(**cfg)
    train_loader, test_loader, val_loader = make_loaders(
        drug,
        expression,
        methylation,
        pairs,
        cell_ids,
        drug_ids,
        batch_size=args.batch_size,
        num_workers=args.workers,
        train_shuffle=False,
    )
    model = BANDRPGate(
        cell_exp_dim=expression.shape[1],
        cell_mut_dim=1,
        cell_meth_dim=methylation.shape[1],
        cell_path_dim=1,
        **cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
    print(f"device={device} pairs={len(pairs)} cells={len(cell_ids)} drugs={len(drug_ids)}")
    fit(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        device,
        args.output_dir,
        args.epochs or cfg.model.epoch,
    )


if __name__ == "__main__":
    main()
