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
        # a,b: [B,d]
        outs = []
        for i in range(self.num_heads):
            ah = self.a_proj[i](a)
            bh = self.b_proj[i](b)
            fh = ah * bh                      # bilinear-like interaction
            fh = self.drop(fh)
            fh = self.out_proj[i](fh)
            outs.append(fh)
        out = torch.stack(outs, dim=0).sum(dim=0)  # [B,d]
        return self.norm(out)


class BANDRPGate(nn.Module):
    """
    4-token Transformer fusion:
      t0: drug_morgan (Protected from dropout)
      t1: cell_exp
      t2: cell_meth
      t3: ban(exp, drug_morgan)
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
        
        # enforce required mods for this stage
        if not (config["mod"]["use_morgan"] and config["mod"]["use_exp"] and config["mod"]["use_meth"]):
            raise ValueError("This 4-token model requires: use_morgan=True, use_exp=True, use_meth=True.")

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

        # token-type + positional embeddings (length fixed = 4)
        self.token_type = nn.Embedding(4, d_model)  # 0=drug,1=exp,2=meth,3=ban
        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

        # [修改] 增加 "gate" 作为合法的 pool 选项
        self.pool = pool
        if self.pool not in ["mean", "cls", "gate"]:
            raise ValueError("trans.pool must be 'mean', 'cls', or 'gate'.")

        # [新增] 动态门控层 (Dynamic Gating)
        # 针对每个样本的每个 token，实时计算其融合权重
        if self.pool == "gate":
            self.modality_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )

        # head
        mlp_hidden_dim = config["mlp"]["mlp_hidden_dim"]
        self.mlp = MLP(d_model, mlp_hidden_dim, out_dim=1, dropout=0.0)

    def forward(self, drug_data, cell_data):
        """
        drug_data: [morgan, espf, pubchem] (only morgan used)
        cell_data: [exp, mut, meth, path] (only exp, meth used)
        """
        morgan = drug_data[0]
        exp = cell_data[0]
        meth = cell_data[2]

        t_drug = self.drug_morgan_enc(morgan)  # [B,d]
        t_exp = self.cell_exp_enc(exp)         # [B,d]
        t_meth = self.cell_meth_enc(meth)      # [B,d]

        # BAN token: interaction between exp and drug token
        t_ban = self.ban_fusion(t_exp, t_drug) # [B,d]

        # stack 4 tokens -> [B,4,d]
        # Order: 0:Drug, 1:Exp, 2:Meth, 3:BAN
        x = torch.stack([t_drug, t_exp, t_meth, t_ban], dim=1)

        # =======================================================
        # [核心修改] Protected Modality Dropout (Token Dropout)
        # =======================================================
        if self.training and self.token_dropout_rate > 0:
            # 1. 基础概率：假设所有Token保留概率都是 (1-p)
            keep_prob = 1 - self.token_dropout_rate
            probs = torch.full((x.shape[0], x.shape[1], 1), keep_prob, device=x.device)
            
            # 2. 【关键保护】强制 Drug Token (Index 0) 的保留概率为 1.0
            probs[:, 0, :] = 1.0
            
            # 3. 采样 Mask (0或1)
            mask = torch.bernoulli(probs)
            
            # 4. 数值缩放 (Inverted Dropout)
            scale_factor = 1.0 / keep_prob
            scales = torch.full_like(probs, scale_factor)
            scales[:, 0, :] = 1.0  # 修正 Drug 的缩放系数为 1.0
            
            x = x * mask * scales
        # =======================================================

        # add type & position
        B = x.size(0)
        tt = torch.tensor([0, 1, 2, 3], device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.token_type(tt) + self.pos

        x = self.transformer(x)  # [B,4,d]

        # [修改] 替换 Pooling 逻辑
        if self.pool == "mean":
            h = x.mean(dim=1)    # [B,d]
        elif self.pool == "cls":
            h = x[:, 0, :]       # [B,d]
        elif self.pool == "gate":
            # [新增] 动态门控池化逻辑
            # 计算每个 Token 的独立打分
            gate_scores = self.modality_gate(x)              # [B, 4, 1]
            # 在 token 维度 (dim=1) 做 Softmax，确保 4 个 token 权重相加为 1
            gate_weights = torch.softmax(gate_scores, dim=1) # [B, 4, 1]
            # 广播相乘并求和，得到自适应加权后的特征
            h = torch.sum(x * gate_weights, dim=1)           # [B, d_model]

        pred = self.mlp(h).squeeze(-1)
        return pred, None


# Backward-compatible symbol for checkpoints and scripts from the original study.
BANDRP = BANDRPGate
