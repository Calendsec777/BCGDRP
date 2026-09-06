#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bcgdrp.data import load_data, make_loaders
from bcgdrp.utils import set_seed
from config import get_cfg_defaults
from model import BANDRP


# -----------------------------
# metrics
# -----------------------------
def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return np.nan
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum())
    if denom == 0:
        return np.nan
    return float((vx * vy).sum() / denom)


def _rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return np.nan
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _safe_pearson(rx, ry)


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    pearson = _safe_pearson(y_true, y_pred)
    spearman = _safe_spearman(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


# -----------------------------
# io helpers
# -----------------------------
def make_eval_loader(original_loader, shuffle=False):
    return DataLoader(
        dataset=original_loader.dataset,
        batch_size=original_loader.batch_size,
        shuffle=shuffle,
        num_workers=original_loader.num_workers,
        drop_last=False,
        collate_fn=original_loader.collate_fn,
        pin_memory=getattr(original_loader, "pin_memory", False),
    )


def write_curve_csv(curve, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_mse",
                "val_mse",
                "test_mse",
            ],
        )
        w.writeheader()
        w.writerows(curve)


def write_json(obj, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_best_txt(
    out_path,
    run_id,
    ablation_mode,
    hparams,
    best_epoch,
    best_val_mse,
    val_metrics,
    test_metrics,
    best_ckpt_path,
):
    lines = []
    lines.append(f"run_id: {run_id}")
    lines.append(f"ablation_mode: {ablation_mode}")
    lines.append("")
    lines.append("hparams:")
    for k in sorted(hparams.keys()):
        lines.append(f"  {k}: {hparams[k]}")

    lines.append("")
    lines.append(f"best_epoch (by val_mse): {best_epoch}")
    lines.append(f"best_val_mse: {best_val_mse:.8f}")
    lines.append(f"best_checkpoint: {best_ckpt_path}")
    lines.append("")

    lines.append("val_metrics_at_best:")
    for k in ["mse", "rmse", "mae", "r2", "pearson", "spearman"]:
        lines.append(f"  {k}: {val_metrics.get(k)}")

    lines.append("")
    lines.append("test_metrics_at_best:")
    for k in ["mse", "rmse", "mae", "r2", "pearson", "spearman"]:
        lines.append(f"  {k}: {test_metrics.get(k)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# train / eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    preds, labels = [], []

    for (drug_data, exp, mut, meth, path, label, _cell_idx) in loader:
        drug_fp = [drug_data[i].to(device) for i in range(len(drug_data))]
        exp = exp.to(device)
        mut = mut.to(device)
        meth = meth.to(device)
        path = path.to(device)
        label = label.to(device)

        pred, _ = model(drug_fp, [exp, mut, meth, path])
        loss = loss_fn(pred, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds.append(pred.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)
    mse = float(np.mean((preds - labels) ** 2))
    return mse


@torch.no_grad()
def evaluate_mse_and_metrics(model, loader, loss_fn, device):
    model.eval()
    preds, labels = [], []

    for (drug_data, exp, mut, meth, path, label, _cell_idx) in loader:
        drug_fp = [drug_data[i].to(device) for i in range(len(drug_data))]
        exp = exp.to(device)
        mut = mut.to(device)
        meth = meth.to(device)
        path = path.to(device)
        label = label.to(device)

        pred, _ = model(drug_fp, [exp, mut, meth, path])

        preds.append(pred.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    labels = np.concatenate(labels).reshape(-1)

    mse = float(np.mean((preds - labels) ** 2))
    metrics = eval_metrics(labels, preds)
    return mse, metrics


# -----------------------------
# one run
# -----------------------------
def run_one_mode(
    cfg,
    mode,
    train_loader,
    val_loader,
    test_loader,
    exp_dim,
    mut_dim,
    meth_dim,
    path_dim,
    device,
    optimizer_name,
    max_epochs,
    early_stop_patience,
    out_dir,
):
    cfg = cfg.clone()
    cfg["trans"]["ablation_mode"] = mode

    run_id = os.path.basename(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = BANDRP(
        cell_exp_dim=exp_dim,
        cell_mut_dim=mut_dim,
        cell_meth_dim=meth_dim,
        cell_path_dim=path_dim,
        **cfg,
    ).to(device)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["model"]["lr"]),
            weight_decay=float(cfg["model"]["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["model"]["lr"]),
            weight_decay=float(cfg["model"]["weight_decay"]),
        )

    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(out_dir, "model_best.pt")
    curve = []

    t0 = time.time()

    print(f"\n[{run_id}] mode={mode}")
    print(
        f"  lr={cfg['model']['lr']} wd={cfg['model']['weight_decay']} "
        f"token_dropout={cfg['trans'].get('token_dropout', 0.0)} "
        f"trans_dropout={cfg['trans']['dropout']} enc_dropout={cfg['enc']['dropout']}"
    )

    for epoch in range(max_epochs):
        train_mse = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mse, _ = evaluate_mse_and_metrics(model, val_loader, loss_fn, device)
        test_mse, _ = evaluate_mse_and_metrics(model, test_loader, loss_fn, device)

        curve.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
            }
        )

        print(
            f"  epoch={epoch:03d} "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} "
            f"test_mse={test_mse:.6f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(f"  early stop triggered at epoch={epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_mse_best, val_metrics_best = evaluate_mse_and_metrics(model, val_loader, loss_fn, device)
    test_mse_best, test_metrics_best = evaluate_mse_and_metrics(model, test_loader, loss_fn, device)

    seconds = time.time() - t0

    write_curve_csv(curve, os.path.join(out_dir, "curve.csv"))
    write_json(dict(cfg), os.path.join(out_dir, "config_used.json"))
    write_best_txt(
        os.path.join(out_dir, "best_result.txt"),
        run_id=run_id,
        ablation_mode=mode,
        hparams={
            "model.lr": float(cfg["model"]["lr"]),
            "model.weight_decay": float(cfg["model"]["weight_decay"]),
            "trans.token_dropout": float(cfg["trans"].get("token_dropout", 0.0)),
            "trans.dropout": float(cfg["trans"]["dropout"]),
            "enc.dropout": float(cfg["enc"]["dropout"]),
            "enc.depth": int(cfg["enc"]["depth"]),
            "trans.num_layers": int(cfg["trans"]["num_layers"]),
            "trans.dim_feedforward": int(cfg["trans"]["dim_feedforward"]),
            "ban.dropout": float(cfg["ban"]["dropout"]),
        },
        best_epoch=best_epoch,
        best_val_mse=best_val_mse,
        val_metrics=val_metrics_best,
        test_metrics=test_metrics_best,
        best_ckpt_path=best_path,
    )

    return {
        "run_id": run_id,
        "ablation_mode": mode,
        "best_epoch": best_epoch,
        "best_val_mse": val_mse_best,
        "best_test_mse": test_mse_best,
        "val_rmse": val_metrics_best["rmse"],
        "val_mae": val_metrics_best["mae"],
        "val_r2": val_metrics_best["r2"],
        "val_pearson": val_metrics_best["pearson"],
        "val_spearman": val_metrics_best["spearman"],
        "test_rmse": test_metrics_best["rmse"],
        "test_mae": test_metrics_best["mae"],
        "test_r2": test_metrics_best["r2"],
        "test_pearson": test_metrics_best["pearson"],
        "test_spearman": test_metrics_best["spearman"],
        "seconds": seconds,
    }


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--early_stop_patience", type=int, default=10)

    parser.add_argument("--out_root", type=str, default=None)

    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "full",
            "only_morgan",
            "only_rnaseq",
            "only_meth",
            "only_ban",
            "rnaseq_morgan",
            "rnaseq_meth",
            "morgan_meth",
            "rnaseq_ban",
            "ban_meth",
            "ban_morgan",
            "no_meth",
            "no_ban",
            "ban_rnaseq_meth",
            "ban_morgan_meth",
            ],
        help="Ablation modes to run",
    )

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--token_dropout", type=float, default=None)
    parser.add_argument("--trans_dropout", type=float, default=None)
    parser.add_argument("--enc_dropout", type=float, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = get_cfg_defaults()

    if args.gpu is not None:
        cfg["model"]["cuda_id"] = args.gpu

    if args.lr is not None:
        cfg["model"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["model"]["weight_decay"] = args.weight_decay
    if args.token_dropout is not None:
        cfg["trans"]["token_dropout"] = args.token_dropout
    if args.trans_dropout is not None:
        cfg["trans"]["dropout"] = args.trans_dropout
    if args.enc_dropout is not None:
        cfg["enc"]["dropout"] = args.enc_dropout

    if args.out_root is None:
        args.out_root = cfg["path"]["savedir"]
    os.makedirs(args.out_root, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg['model']['cuda_id']}")
    else:
        device = torch.device("cpu")
    print(f"[Device] Running on {device}")

    drug_feature, exp_feature, methy_feature, pair, depmap_id, drug_id = load_data(**cfg)

    train_loader, test_loader, val_loader = make_loaders(
        drug_feature,
        exp_feature,
        methy_feature,
        pair,
        depmap_id,
        drug_id,
    )

    val_loader = make_eval_loader(val_loader, shuffle=False)
    test_loader = make_eval_loader(test_loader, shuffle=False)

    exp_dim = exp_feature.shape[-1]
    mut_dim = 1
    meth_dim = methy_feature.shape[-1]
    path_dim = 1

    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(cfg["model"]["epoch"])

    summary_csv = os.path.join(args.out_root, "ablation_summary.csv")
    summary_fields = [
        "run_id",
        "ablation_mode",
        "best_epoch",
        "best_val_mse",
        "best_test_mse",
        "val_rmse",
        "val_mae",
        "val_r2",
        "val_pearson",
        "val_spearman",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_pearson",
        "test_spearman",
        "seconds",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()

    all_results = []
    for idx, mode in enumerate(args.modes):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"abl_{idx:02d}_{mode}_{ts}"
        out_dir = os.path.join(args.out_root, run_id)

        result = run_one_mode(
            cfg=cfg,
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dim=exp_dim,
            mut_dim=mut_dim,
            meth_dim=meth_dim,
            path_dim=path_dim,
            device=device,
            optimizer_name=args.optimizer,
            max_epochs=max_epochs,
            early_stop_patience=args.early_stop_patience,
            out_dir=out_dir,
        )

        all_results.append(result)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writerow(result)

    ranked_csv = os.path.join(args.out_root, "ablation_summary_ranked.csv")
    ranked_results = sorted(all_results, key=lambda x: x["best_val_mse"])
    with open(ranked_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(ranked_results)

    print("\n===== Ablation done =====")
    print(f"Summary saved to: {summary_csv}")
    print(f"Ranked summary saved to: {ranked_csv}")
    print("Top results by val_mse:")
    for r in ranked_results[:5]:
        print(
            f"  {r['ablation_mode']:>16s} | "
            f"val_mse={r['best_val_mse']:.6f} | "
            f"test_mse={r['best_test_mse']:.6f} | "
            f"test_pearson={r['test_pearson']:.4f}"
        )


if __name__ == "__main__":
    main()
