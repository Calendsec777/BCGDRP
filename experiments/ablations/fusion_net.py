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


class ConcatMLPFusion(nn.Module):
    def __init__(self, d_model, hidden_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * hidden_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=-1)
        return self.out_norm(self.net(x))


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, ff_ratio=2):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(d_model * ff_ratio)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, a, b):
        # a queries b, output is one fused vector token
        q = self.q_norm(a).unsqueeze(1)   # [B,1,d]
        kv = self.kv_norm(b).unsqueeze(1) # [B,1,d]
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = attn_out.squeeze(1)
        h = h + self.ff(h)
        return self.out_norm(h)


class FusionFactory(nn.Module):
    def __init__(self, d_model, fusion_cfg):
        super().__init__()
        fusion_type = str(fusion_cfg["type"]).lower()
        self.fusion_type = fusion_type
        self.none_mode = str(fusion_cfg.get("none_mode", "zero")).lower()

        if fusion_type == "ban":
            self.fuser = BANFusion(
                d_model=d_model,
                num_heads=int(fusion_cfg.get("ban_num_heads", 4)),
                dropout=float(fusion_cfg.get("ban_dropout", 0.1)),
            )
        elif fusion_type == "concat_mlp":
            self.fuser = ConcatMLPFusion(
                d_model=d_model,
                hidden_ratio=float(fusion_cfg.get("concat_hidden_ratio", 2)),
                dropout=float(fusion_cfg.get("concat_dropout", 0.1)),
            )
        elif fusion_type == "cross_attn":
            self.fuser = CrossAttentionFusion(
                d_model=d_model,
                num_heads=int(fusion_cfg.get("cross_num_heads", 4)),
                dropout=float(fusion_cfg.get("cross_dropout", 0.1)),
                ff_ratio=float(fusion_cfg.get("cross_ff_ratio", 2)),
            )
        elif fusion_type == "none":
            self.fuser = None
        else:
            raise ValueError(f"Unsupported fusion.type={fusion_type}")

    def forward(self, a, b):
        if self.fusion_type == "none":
            if self.none_mode == "zero":
                return torch.zeros_like(a)
            raise ValueError(f"Unsupported fusion.none_mode={self.none_mode}")
        return self.fuser(a, b)


class BANDRPFusionCompare(nn.Module):
    """
    Fixed 4-token transformer input:
      t0: drug_morgan
      t1: cell_exp
      t2: cell_meth
      t3: fusion token (BAN / concat+MLP / cross-attention / none)

    fusion.pair:
      - exp_drug  => fusion(t_exp,  t_drug)
      - meth_drug => fusion(t_meth, t_drug)
    """
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, **config):
        super().__init__()

        d_model = int(config["trans"]["d_model"])
        nhead = int(config["trans"]["nhead"])
        num_layers = int(config["trans"]["num_layers"])
        dim_ff = int(config["trans"]["dim_feedforward"])
        trans_dropout = float(config["trans"]["dropout"])
        pool = str(config["trans"]["pool"])

        enc_depth = int(config["enc"]["depth"])
        enc_ratio = int(config["enc"]["mlp_ratio"])
        enc_dropout = float(config["enc"]["dropout"])

        self.token_dropout_rate = float(config["trans"].get("token_dropout", 0.0))
        self.fusion_type = str(config["fusion"].get("type", "ban")).lower()
        self.fusion_pair = str(config["fusion"].get("pair", "exp_drug")).lower()

        if self.fusion_pair not in ["exp_drug", "meth_drug"]:
            raise ValueError("fusion.pair must be 'exp_drug' or 'meth_drug'.")

        if not (config["mod"].get("use_morgan", True) and config["mod"].get("use_exp", True) and config["mod"].get("use_meth", True)):
            raise ValueError("This fusion-comparison model requires use_morgan=True, use_exp=True, use_meth=True.")

        self.drug_morgan_enc = ResMLPEncoder(
            in_dim=2048, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )
        self.cell_exp_enc = ResMLPEncoder(
            in_dim=cell_exp_dim, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )
        self.cell_meth_enc = ResMLPEncoder(
            in_dim=cell_meth_dim, d_model=d_model, depth=enc_depth, mlp_ratio=enc_ratio, dropout=enc_dropout
        )

        self.fusion = FusionFactory(d_model=d_model, fusion_cfg=config["fusion"])

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=trans_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.token_type = nn.Embedding(4, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        self.pool = pool
        if self.pool not in ["mean", "cls"]:
            raise ValueError("trans.pool must be 'mean' or 'cls'.")

        mlp_hidden_dim = config["mlp"]["mlp_hidden_dim"]
        self.mlp = MLP(d_model, mlp_hidden_dim, out_dim=1, dropout=0.0)

        print(f"[FusionCompare] fusion_type = {self.fusion_type}")
        print(f"[FusionCompare] fusion_pair = {self.fusion_pair}")
        print(f"[FusionCompare] token_dropout_rate = {self.token_dropout_rate}")

    def _build_tokens(self, drug_data, cell_data):
        morgan = drug_data[0]
        exp = cell_data[0]
        meth = cell_data[2]

        t_drug = self.drug_morgan_enc(morgan)
        t_exp = self.cell_exp_enc(exp)
        t_meth = self.cell_meth_enc(meth)

        if self.fusion_pair == "exp_drug":
            t_fusion = self.fusion(t_exp, t_drug)
        else:
            t_fusion = self.fusion(t_meth, t_drug)

        return t_drug, t_exp, t_meth, t_fusion

    def _apply_token_dropout(self, x):
        if (not self.training) or self.token_dropout_rate <= 0:
            return x

        keep_prob = 1.0 - self.token_dropout_rate
        probs = torch.full((x.shape[0], x.shape[1], 1), keep_prob, device=x.device)
        probs[:, 0, :] = 1.0  # always protect drug token

        mask = torch.bernoulli(probs)
        scales = torch.full_like(probs, 1.0 / keep_prob)
        scales[:, 0, :] = 1.0
        return x * mask * scales

    def forward(self, drug_data, cell_data):
        t_drug, t_exp, t_meth, t_fusion = self._build_tokens(drug_data, cell_data)
        x = torch.stack([t_drug, t_exp, t_meth, t_fusion], dim=1)  # [B,4,d]
        x = self._apply_token_dropout(x)

        B = x.size(0)
        tt = torch.tensor([0, 1, 2, 3], device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.token_type(tt) + self.pos

        x = self.transformer(x)

        if self.pool == "mean":
            h = x.mean(dim=1)
        else:
            h = x[:, 0, :]

        pred = self.mlp(h).squeeze(-1)
        aux = {
            "fusion_type": self.fusion_type,
            "fusion_pair": self.fusion_pair,
        }
        return pred, aux
