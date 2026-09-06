#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _infonce_diag(logits: torch.Tensor) -> torch.Tensor:
    """
    Standard InfoNCE with diagonal positives.
    logits: [B,B]
    loss = (CE(logits, arange(B)) + CE(logits.T, arange(B)))/2
    """
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class ProjectionHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear"""
    def __init__(self, in_dim: int, out_dim: int = 128, hidden_dim=None, dropout: float = 0.0):
        super().__init__()
        h = int(hidden_dim) if hidden_dim is not None else in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BANFuseAlign(nn.Module):
    """
    Align BAN token with fused target z_f(drug+cell) via InfoNCE.

    Inputs:
      t_ban  : [B, d_model]  (BAN token)
      t_drug : [B, d_model]  (Morgan token)
      t_cell : [B, d_model]  (typically exp token; you也可以改成 exp+meth 的某种融合)

    Steps:
      z_b = norm(Proj_ban(t_ban))
      z_d = norm(Proj_d(t_drug))
      z_c = norm(Proj_c(t_cell))
      z_f = norm(z_d + z_c)  or gated: norm(g*z_d + (1-g)*z_c)

      logits = z_b @ z_f^T / tau
      loss   = InfoNCE (diag positives), optionally symmetric (we用的是 symmetric)
    """
    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        temperature: float = 0.1,
        proj_dropout: float = 0.1,
        use_gate: bool = False,
        stopgrad_fuse: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau = float(temperature)
        self.use_gate = bool(use_gate)
        self.stopgrad_fuse = bool(stopgrad_fuse)
        self.eps = float(eps)

        self.proj_b = ProjectionHead(d_model, out_dim=proj_dim, hidden_dim=d_model, dropout=proj_dropout)
        self.proj_d = ProjectionHead(d_model, out_dim=proj_dim, hidden_dim=d_model, dropout=proj_dropout)
        self.proj_c = ProjectionHead(d_model, out_dim=proj_dim, hidden_dim=d_model, dropout=proj_dropout)

        if self.use_gate:
            # gate on concatenated projected (before norm) vectors
            self.gate = nn.Sequential(
                nn.Linear(2 * proj_dim, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, 1),
            )

    def forward(self, t_ban: torch.Tensor, t_drug: torch.Tensor, t_cell: torch.Tensor) -> torch.Tensor:
        if self.tau <= 0:
            raise ValueError("temperature must be > 0")

        zb = _safe_normalize(self.proj_b(t_ban), eps=self.eps)
        zd = _safe_normalize(self.proj_d(t_drug), eps=self.eps)
        zc = _safe_normalize(self.proj_c(t_cell), eps=self.eps)

        if not self.use_gate:
            zf = _safe_normalize(zd + zc, eps=self.eps)
        else:
            g = torch.sigmoid(self.gate(torch.cat([zd, zc], dim=-1)))  # [B,1]
            zf = _safe_normalize(g * zd + (1.0 - g) * zc, eps=self.eps)

        if self.stopgrad_fuse:
            zf = zf.detach()

        logits = (zb @ zf.t()) / self.tau
        return _infonce_diag(logits)
