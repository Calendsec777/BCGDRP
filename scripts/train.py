#!/usr/bin/env python3
"""Train the final BCGDRP model on the manuscript cell-line-disjoint split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bcgdrp.cell_loss import ExpMethContrastive  # noqa: E402
from bcgdrp.config import configure_paths, load_config  # noqa: E402
from bcgdrp.data import load_data, make_loaders  # noqa: E402
from bcgdrp.fusion_loss import BANFuseAlign  # noqa: E402
from bcgdrp.model import BCGDRP  # noqa: E402
from bcgdrp.training import fit  # noqa: E402
from bcgdrp.utils import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "bcgdrp.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "bcgdrp")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
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
        train_shuffle=True,
    )

    model = BCGDRP(
        cell_exp_dim=expression.shape[1],
        cell_mut_dim=1,
        cell_meth_dim=methylation.shape[1],
        cell_path_dim=1,
        **cfg,
    ).to(device)

    losses = {}
    if cfg.cl.use_exp_meth:
        losses["exp_meth"] = ExpMethContrastive(
            cfg.trans.d_model,
            cfg.cl.proj_dim,
            cfg.cl.temp_cell,
            detach_other=cfg.cl.detach_other,
            proj_dropout=cfg.cl.proj_dropout,
        ).to(device)
    if cfg.cl.use_exp_morgan:
        losses["exp_morgan"] = ExpMethContrastive(
            cfg.trans.d_model,
            cfg.cl.proj_dim,
            cfg.cl.temp_cross,
            detach_other=cfg.cl.detach_other,
            proj_dropout=cfg.cl.proj_dropout,
        ).to(device)
    if cfg.cl.use_ban_fuse:
        losses["ban_fuse"] = BANFuseAlign(
            cfg.trans.d_model,
            cfg.cl.proj_dim,
            cfg.cl.temp_ban,
            proj_dropout=cfg.cl.proj_dropout,
            use_gate=cfg.cl.ban_gate,
        ).to(device)

    parameters = list(model.parameters())
    for loss_module in losses.values():
        parameters.extend(loss_module.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)
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
        cfg=cfg,
        losses=losses,
    )


if __name__ == "__main__":
    main()
