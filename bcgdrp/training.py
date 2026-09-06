"""Training and evaluation helpers shared by the two released models."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else math.nan


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        if end > start:
            ranks[order[start : end + 1]] = (start + end + 2) / 2.0
        start = end + 1
    return ranks


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the regression metrics reported in the manuscript."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    mse = float(np.mean(residual**2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total else math.nan,
        "pearson": _pearson(y_true, y_pred),
        "spearman": _pearson(_rank(y_true), _rank(y_pred)),
    }


def _to_device(batch, device: torch.device):
    drug_data, expression, mutation, methylation, pathway, labels, cell_idx = batch
    drug_data = [feature.to(device) for feature in drug_data]
    cell_data = [
        expression.to(device),
        mutation.to(device),
        methylation.to(device),
        pathway.to(device),
    ]
    return drug_data, cell_data, labels.to(device), cell_idx.to(device)


def train_epoch(model, loader, optimizer, device, cfg=None, losses=None) -> float:
    model.train()
    losses = losses or {}
    predictions, labels_all = [], []
    loss_fn = torch.nn.MSELoss()

    for batch in loader:
        drug_data, cell_data, labels, cell_idx = _to_device(batch, device)
        predictions_batch, aux = model(drug_data, cell_data)
        loss = loss_fn(predictions_batch, labels)

        if cfg is not None and aux is not None:
            if cfg.cl.use_exp_meth and "exp_meth" in losses:
                loss = loss + cfg.cl.lambda_cell * losses["exp_meth"](
                    aux["t_exp_g"], aux["t_meth_g"], cell_idx=cell_idx
                )
            if cfg.cl.use_exp_morgan and "exp_morgan" in losses:
                loss = loss + cfg.cl.lambda_cross * losses["exp_morgan"](
                    aux["t_exp_g"], aux["t_drug"]
                )
            if cfg.cl.use_ban_fuse and "ban_fuse" in losses:
                loss = loss + cfg.cl.lambda_ban * losses["ban_fuse"](
                    aux["t_ban"], aux["t_drug"], aux["t_exp_g"]
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        predictions.append(predictions_batch.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    return metrics(np.concatenate(labels_all), np.concatenate(predictions))["mse"]


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    predictions, labels_all = [], []
    for batch in loader:
        drug_data, cell_data, labels, _ = _to_device(batch, device)
        predictions_batch, _ = model(drug_data, cell_data)
        predictions.append(predictions_batch.cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    return metrics(np.concatenate(labels_all), np.concatenate(predictions))


def fit(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    device,
    output_dir: str | Path,
    epochs: int,
    *,
    cfg=None,
    losses=None,
) -> dict:
    """Train one model and retain the checkpoint selected by validation MSE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    curve = []
    best_epoch = -1
    best_val = math.inf
    started = time.time()

    for epoch in range(epochs):
        train_mse = train_epoch(model, train_loader, optimizer, device, cfg, losses)
        val = evaluate(model, val_loader, device)
        test = evaluate(model, test_loader, device)
        curve.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val["mse"], "test_mse": test["mse"]})
        print(
            f"epoch={epoch:03d} train_mse={train_mse:.6f} "
            f"val_mse={val['mse']:.6f} test_mse={test['mse']:.6f}"
        )
        if val["mse"] < best_val:
            best_val = val["mse"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "val_mse", "test_mse"])
        writer.writeheader()
        writer.writerows(curve)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    summary = {
        "best_epoch": best_epoch,
        "seconds": time.time() - started,
        "validation": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
        "checkpoint": str(checkpoint),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
