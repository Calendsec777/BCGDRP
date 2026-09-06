#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contrastive alignment for cell embeddings: RNAseq (exp) ↔ methylation (meth)

适用于 DRP / drug–cell pair 的 batch：同一个 cell 可能在一个 batch 里出现多次（对应不同 drug）。
因此支持传入 cell_idx，把“同 cell”的样本都视为 positives，避免 false negatives。

最小用法（training step）：
    from contrastive_exp_meth import exp_meth_contrastive_loss

    # t_exp, t_meth: [B, D]（建议是进入 Transformer 前的 token embedding）
    loss_reg = mse(pred, y)
    loss_cl  = exp_meth_contrastive_loss(t_exp, t_meth, cell_idx=batch_cell_idx,
                                         temperature=0.1, symmetric=True)
    loss = loss_reg + lambda_cl * loss_cl
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def _make_pos_mask_from_cell_idx(cell_idx: torch.Tensor) -> torch.Tensor:
    """
    cell_idx: [B] (int/long).
    返回 pos_mask: [B,B] bool，其中 pos_mask[i,j]=True 当且仅当 cell_idx[i] == cell_idx[j]
    """
    if cell_idx.dim() != 1:
        raise ValueError(f"cell_idx must be 1D [B], got shape={tuple(cell_idx.shape)}")
    return (cell_idx.view(-1, 1) == cell_idx.view(1, -1))


def _infonce_multi_positive(logits: torch.Tensor, pos_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Multi-positive InfoNCE（每个 anchor 可以有多个正样本）:
      loss_i = -log( sum_{j in P(i)} exp(logits_ij) / sum_{k} exp(logits_ik) )

    logits:   [B,B]
    pos_mask: [B,B] bool
    """
    if logits.shape != pos_mask.shape:
        raise ValueError(f"logits and pos_mask must have same shape, got {logits.shape} vs {pos_mask.shape}")

    # 数值稳定：每行减 max
    logits = logits - logits.max(dim=1, keepdim=True).values

    exp_logits = torch.exp(logits)
    pos = exp_logits * pos_mask.to(exp_logits.dtype)

    sum_pos = pos.sum(dim=1)        # [B]
    sum_all = exp_logits.sum(dim=1) # [B]

    valid = sum_pos > 0
    if valid.sum() == 0:
        # 理论上不会发生（至少对角线应为 True），但保底防 NaN
        return logits.new_tensor(0.0)

    loss = -torch.log((sum_pos[valid] + eps) / (sum_all[valid] + eps))
    return loss.mean()


def exp_meth_contrastive_loss(
    z_exp: torch.Tensor,
    z_meth: torch.Tensor,
    cell_idx: torch.Tensor | None = None,
    temperature: float = 0.1,
    symmetric: bool = True,
    detach_other: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    CLIP-style exp ↔ meth 对比学习损失（InfoNCE）。

    Args:
        z_exp, z_meth: [B,D] embedding（建议：Transformer 之前的 token embedding）
        cell_idx: 可选 [B] long。若提供：同一个 cell 的样本两两都当 positives
        temperature: tau
        symmetric: True 时返回 0.5*(exp->meth + meth->exp)
        detach_other: True 时在每个方向上 detach target 侧，减少“互相拉扯”
        eps: normalize 数值稳定

    Returns:
        scalar loss
    """
    if z_exp.dim() != 2 or z_meth.dim() != 2:
        raise ValueError(f"z_exp and z_meth must be 2D [B,D], got {z_exp.shape} and {z_meth.shape}")
    if z_exp.shape[0] != z_meth.shape[0]:
        raise ValueError(f"Batch size mismatch: {z_exp.shape[0]} vs {z_meth.shape[0]}")

    B = z_exp.shape[0]
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("temperature must be > 0")

    # L2 normalize
    ze = _safe_normalize(z_exp, dim=-1, eps=eps)
    zm = _safe_normalize(z_meth, dim=-1, eps=eps)

    # positives mask
    if cell_idx is None:
        pos_mask = torch.eye(B, device=ze.device, dtype=torch.bool)
    else:
        if cell_idx.device != ze.device:
            cell_idx = cell_idx.to(ze.device)
        pos_mask = _make_pos_mask_from_cell_idx(cell_idx)

    # exp -> meth
    zm_tgt = zm.detach() if detach_other else zm
    logits_em = (ze @ zm_tgt.t()) / tau
    loss_em = _infonce_multi_positive(logits_em, pos_mask)

    if not symmetric:
        return loss_em

    # meth -> exp
    ze_tgt = ze.detach() if detach_other else ze
    logits_me = (zm @ ze_tgt.t()) / tau
    loss_me = _infonce_multi_positive(logits_me, pos_mask)

    return 0.5 * (loss_em + loss_me)


# -----------------------------
# 可选：投影头 + 模块封装（更方便集成到 model 里）
# -----------------------------

class ProjectionHead(nn.Module):
    """小投影头：Linear -> GELU -> Dropout -> Linear"""
    def __init__(self, in_dim: int, out_dim: int = 128, hidden_dim: int | None = None, dropout: float = 0.0):
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


class ExpMethContrastive(nn.Module):
    """
    exp↔meth 对比损失模块：可选择是否加 projection head
    """
    def __init__(
        self,
        d_model: int,
        proj_dim: int = 128,
        temperature: float = 0.1,
        symmetric: bool = True,
        detach_other: bool = False,
        proj_dropout: float = 0.0,
        use_proj: bool = True,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.detach_other = bool(detach_other)
        self.use_proj = bool(use_proj)

        if self.use_proj:
            self.proj_exp = ProjectionHead(d_model, out_dim=proj_dim, hidden_dim=d_model, dropout=proj_dropout)
            self.proj_meth = ProjectionHead(d_model, out_dim=proj_dim, hidden_dim=d_model, dropout=proj_dropout)
        else:
            self.proj_exp = None
            self.proj_meth = None

    def forward(self, t_exp: torch.Tensor, t_meth: torch.Tensor, cell_idx: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_proj:
            z_exp = self.proj_exp(t_exp)
            z_meth = self.proj_meth(t_meth)
        else:
            z_exp, z_meth = t_exp, t_meth

        return exp_meth_contrastive_loss(
            z_exp=z_exp,
            z_meth=z_meth,
            cell_idx=cell_idx,
            temperature=self.temperature,
            symmetric=self.symmetric,
            detach_other=self.detach_other,
        )
