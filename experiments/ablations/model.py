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
    Vector -> d_model with residual MLP blocks:
    proj + N*(LN->Linear->GELU->Dropout->Linear->Dropout + residual) + LN
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
    """
    Multi-head bilinear fusion that outputs ONE vector token.
    Inputs: a,b in R^{Bxd}
    For each head: f_h = W_a(a) ⊙ W_b(b), then proj back to d, sum heads, LN.
    """
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
        out = torch.stack(outs, dim=0).sum(dim=0)  # [B,d]
        return self.norm(out)


class BANDRP(nn.Module):
    """
    Universal ablation-ready Transformer fusion.

    Available base tokens:
      0: drug_morgan
      1: cell_exp (RNA-seq)
      2: cell_meth
      3: ban(exp, drug_morgan)

    Supported ablation modes:
      - full
      - only_morgan
      - only_rnaseq
      - only_meth
      - only_ban
      - rnaseq_morgan
      - rnaseq_meth
      - morgan_meth
      - rnaseq_ban
      - ban_meth
      - ban_morgan
      - no_meth          (= morgan + rnaseq + ban)
      - no_ban           (= morgan + rnaseq + meth)
      - ban_rnaseq_meth  (= ban + rnaseq + meth)
      - ban_morgan_meth  (= ban + morgan + meth)
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

        self.token_dropout_rate = float(config["trans"].get("token_dropout", 0.0))
        self.ablation_mode = str(config["trans"].get("ablation_mode", "full")).lower()

        # 这里不再强制 full model 三模态全开
        # 但由于 BAN 仍依赖 morgan 和 exp，因此相关 encoder 仍保留
        self.use_morgan = bool(config["mod"].get("use_morgan", True))
        self.use_exp = bool(config["mod"].get("use_exp", True))
        self.use_meth = bool(config["mod"].get("use_meth", True))

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

        # --- BAN token ---
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

        # max token number = 4, but actual token length is dynamic
        self.token_type = nn.Embedding(4, d_model)  # 0=drug,1=exp,2=meth,3=ban
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.pool = pool
        if self.pool not in ["mean", "cls"]:
            raise ValueError("trans.pool must be 'mean' or 'cls'.")

        mlp_hidden_dim = config["mlp"]["mlp_hidden_dim"]
        self.mlp = MLP(d_model, mlp_hidden_dim, out_dim=1, dropout=0.0)

        self.mode_to_token_ids = {
            "full": [0, 1, 2, 3],
            "only_morgan": [0],
            "only_rnaseq": [1],
            "only_meth": [2],
            "only_ban": [3],
            "rnaseq_morgan": [0, 1],
            "rnaseq_meth": [1, 2],
            "morgan_meth": [0, 2],
            "rnaseq_ban": [1, 3],
            "ban_meth": [2, 3],
            "ban_morgan": [0, 3],
            "no_meth": [0, 1, 3],          # morgan + rnaseq + ban
            "no_ban": [0, 1, 2],           # morgan + rnaseq + meth
            "ban_rnaseq_meth": [1, 2, 3],  # rnaseq + meth + ban
            "ban_morgan_meth": [0, 2, 3],  # morgan + meth + ban
        }

        if self.ablation_mode not in self.mode_to_token_ids:
            raise ValueError(
                f"Unknown ablation_mode={self.ablation_mode}. "
                f"Supported: {list(self.mode_to_token_ids.keys())}"
            )

        # 若你确实关闭了对应模态开关，这里做一致性检查
        self._check_ablation_compatibility()

        print(f"[BANDRP] ablation_mode = {self.ablation_mode}")
        print(f"[BANDRP] token_dropout_rate = {self.token_dropout_rate}")

    def _check_ablation_compatibility(self):
        token_ids = self.mode_to_token_ids[self.ablation_mode]

        # token 0 = morgan
        if 0 in token_ids and not self.use_morgan:
            raise ValueError(f"ablation_mode={self.ablation_mode} requires use_morgan=True")

        # token 1 = exp
        if 1 in token_ids and not self.use_exp:
            raise ValueError(f"ablation_mode={self.ablation_mode} requires use_exp=True")

        # token 2 = meth
        if 2 in token_ids and not self.use_meth:
            raise ValueError(f"ablation_mode={self.ablation_mode} requires use_meth=True")

        # token 3 = BAN，需要 morgan + exp
        if 3 in token_ids:
            if not self.use_morgan:
                raise ValueError(f"ablation_mode={self.ablation_mode} requires use_morgan=True for BAN")
            if not self.use_exp:
                raise ValueError(f"ablation_mode={self.ablation_mode} requires use_exp=True for BAN")

    def _build_all_tokens(self, drug_data, cell_data):
        """
        drug_data: [morgan, espf, pubchem]
        cell_data: [exp, mut, meth, path]
        """
        morgan = drug_data[0]
        exp = cell_data[0]
        meth = cell_data[2]

        t_drug = self.drug_morgan_enc(morgan)  # [B,d]
        t_exp = self.cell_exp_enc(exp)         # [B,d]
        t_meth = self.cell_meth_enc(meth)      # [B,d]
        t_ban = self.ban_fusion(t_exp, t_drug) # [B,d]

        return {
            0: t_drug,
            1: t_exp,
            2: t_meth,
            3: t_ban,
        }

    def _select_tokens(self, token_dict):
        token_ids = self.mode_to_token_ids[self.ablation_mode]
        x = torch.stack([token_dict[i] for i in token_ids], dim=1)  # [B,L,d]
        return x, token_ids

    def _apply_token_dropout(self, x, token_ids):
        """
        Protected token dropout:
        - protect morgan token (token_type=0) if present
        - for other kept tokens, apply inverted dropout
        """
        if (not self.training) or self.token_dropout_rate <= 0:
            return x

        keep_prob = 1.0 - self.token_dropout_rate
        B, L, _ = x.shape

        probs = torch.full((B, L, 1), keep_prob, device=x.device)

        # protect morgan token if present in this ablation mode
        for j, tid in enumerate(token_ids):
            if tid == 0:
                probs[:, j, :] = 1.0

        mask = torch.bernoulli(probs)

        scales = torch.full_like(probs, 1.0 / keep_prob)
        for j, tid in enumerate(token_ids):
            if tid == 0:
                scales[:, j, :] = 1.0

        x = x * mask * scales
        return x

    def forward(self, drug_data, cell_data):
        token_dict = self._build_all_tokens(drug_data, cell_data)
        x, token_ids = self._select_tokens(token_dict)   # [B,L,d]

        x = self._apply_token_dropout(x, token_ids)

        B, L, _ = x.shape
        tt = torch.tensor(token_ids, device=x.device).unsqueeze(0).expand(B, -1)  # [B,L]
        x = x + self.token_type(tt) + self.pos[:, :L, :]

        x = self.transformer(x)  # [B,L,d]

        if self.pool == "mean":
            h = x.mean(dim=1)
        else:
            h = x[:, 0, :]

        pred = self.mlp(h).squeeze(-1)

        aux = {
            "ablation_mode": self.ablation_mode,
            "token_ids": token_ids,
        }
        return pred, aux