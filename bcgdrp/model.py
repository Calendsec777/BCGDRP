#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResMLPEncoder(nn.Module):
    """
    Vector -> d_model with residual MLP blocks
    """
    def __init__(self, in_dim, d_model, depth=2, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * mlp_ratio, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_norm(h)


class BANFusion(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.a_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_heads)])
        self.b_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_heads)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_heads)])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, a, b):
        outs = []
        for i in range(self.num_heads):
            ah = self.a_proj[i](a)
            bh = self.b_proj[i](b)
            fh = ah * bh
            fh = self.drop(fh)
            fh = self.out_proj[i](fh)
            outs.append(fh)
        out = torch.stack(outs, dim=0).sum(dim=0)
        return self.norm(out)


class DrugConditionedScalarGate(nn.Module):
    """
    Weak scalar gate:
      g = sigmoid(MLP(t_drug))  -> [B,1]
    """
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, t_drug):
        return self.net(t_drug)  # [B,1]


class BCGDRP(nn.Module):
    """
    4-token Transformer fusion + Contrastive Learning + Modified DCG

    t0: drug_morgan (protected)
    t1: gated exp token
    t2: gated/ungated meth token
    t3: BAN(gated_exp, drug)
    """
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, **config):
        super().__init__()

        # --- configs ---
        d_model = int(config["trans"]["d_model"])
        nhead = int(config["trans"]["nhead"])
        num_layers = int(config["trans"]["num_layers"])
        dim_ff = int(config["trans"]["dim_feedforward"])
        trans_dropout = float(config["trans"]["dropout"])
        pool = str(config["trans"]["pool"])

        enc_depth = int(config["enc"]["depth"])
        enc_ratio = int(config["enc"]["mlp_ratio"])
        enc_dropout = float(config["enc"]["dropout"])

        ban_heads = int(config["ban"]["num_heads"])
        ban_dropout = float(config["ban"]["dropout"])

        self.token_dropout_rate = float(
            config["trans"].get("token_dropout", config["ban"].get("token_dropout", 0.0))
        )

        if not (config["mod"]["use_morgan"] and config["mod"]["use_exp"] and config["mod"]["use_meth"]):
            raise ValueError("This model requires: use_morgan=True, use_exp=True, use_meth=True.")

        # --- encoders ---
        self.drug_morgan_enc = ResMLPEncoder(
            in_dim=2048, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )
        self.cell_exp_enc = ResMLPEncoder(
            in_dim=cell_exp_dim, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )
        self.cell_meth_enc = ResMLPEncoder(
            in_dim=cell_meth_dim, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )

        # --- modified DCG ---
        self.use_dcg = bool(config.get("dcg", {}).get("enable", False))
        self.dcg_alpha = float(config.get("dcg", {}).get("alpha", 0.5))
        self.dcg_target = str(config.get("dcg", {}).get("target", "exp"))  # "exp", "exp_meth"

        self.exp_gate = DrugConditionedScalarGate(d_model)
        self.meth_gate = DrugConditionedScalarGate(d_model)

        # --- BAN ---
        self.ban_fusion = BANFusion(d_model=d_model, num_heads=ban_heads, dropout=ban_dropout)

        # --- transformer ---
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=trans_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # --- embeddings ---
        self.token_type = nn.Embedding(4, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.pool = pool
        if self.pool not in ["mean", "cls"]:
            raise ValueError("trans.pool must be 'mean' or 'cls'.")

        # --- head ---
        mlp_hidden_dim = config["mlp"]["mlp_hidden_dim"]
        self.mlp = MLP(d_model, mlp_hidden_dim, out_dim=1, dropout=0.0)

    def forward(self, drug_data, cell_data):
        morgan = drug_data[0]
        exp = cell_data[0]
        meth = cell_data[2]

        # base tokens
        t_drug = self.drug_morgan_enc(morgan)   # [B,d]
        t_exp = self.cell_exp_enc(exp)          # [B,d]
        t_meth = self.cell_meth_enc(meth)       # [B,d]

        # -----------------------------
        # Modified DCG (weak scalar)
        # -----------------------------
        g_exp = None
        g_meth = None

        t_exp_g = t_exp
        t_meth_g = t_meth

        if self.use_dcg:
            if self.dcg_target in ["exp", "exp_meth"]:
                g_exp = self.exp_gate(t_drug)  # [B,1]
                t_exp_g = t_exp * (1.0 + self.dcg_alpha * g_exp)

            if self.dcg_target == "exp_meth":
                g_meth = self.meth_gate(t_drug)  # [B,1]
                t_meth_g = t_meth * (1.0 + self.dcg_alpha * g_meth)

        # BAN uses gated exp
        t_ban = self.ban_fusion(t_exp_g, t_drug)

        # 4 tokens
        x = torch.stack([t_drug, t_exp_g, t_meth_g, t_ban], dim=1)

        # protected token dropout
        if self.training and self.token_dropout_rate > 0:
            probs = torch.full((x.shape[0], x.shape[1], 1), 1 - self.token_dropout_rate, device=x.device)
            probs[:, 0, :] = 1.0  # protect drug
            mask = torch.bernoulli(probs)

            scale = 1.0 / (1 - self.token_dropout_rate)
            scales = torch.full_like(probs, scale)
            scales[:, 0, :] = 1.0
            x = x * mask * scales

        # token type + pos
        B = x.size(0)
        tt = torch.tensor([0, 1, 2, 3], device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.token_type(tt) + self.pos

        # transformer
        x = self.transformer(x)

        if self.pool == "mean":
            h = x.mean(dim=1)
        else:
            h = x[:, 0, :]

        pred = self.mlp(h).squeeze(-1)

        # aux for CL
        aux = {
            "t_drug": t_drug,
            "t_exp": t_exp,          # raw exp token
            "t_meth": t_meth,        # raw meth token
            "t_exp_g": t_exp_g,      # gated exp token
            "t_meth_g": t_meth_g,    # gated meth token
            "t_ban": t_ban,
            "h_fused": h,
            "g_exp": g_exp,
            "g_meth": g_meth,
        }

        return pred, aux


# Backward-compatible symbol for checkpoints and scripts from the original study.
BANDRP = BCGDRP
