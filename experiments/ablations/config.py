from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

# =========================
# Path config
# =========================
_C.path = CN()

ROOT = Path(__file__).resolve().parents[2]

# 默认输出目录
_C.path.savedir = str(ROOT / "outputs" / "ablations")

# response + cell features
_C.path.response = str(ROOT / "data" / "GDSC2_IC50.csv")
_C.path.methylation = str(ROOT / "data" / "cell" / "geo_methylation_cosmic.csv")
_C.path.expression = str(ROOT / "data" / "cell" / "geo_expression_cosmic.csv")

# drug features
_C.path.morgan = str(ROOT / "data" / "drug" / "morgan_encoding.pkl")

# =========================
# Training config
# =========================
_C.model = CN()
_C.model.cuda_id = 1
_C.model.epoch = 80
_C.model.lr = 1e-3
_C.model.weight_decay = 1e-4

# =========================
# Modality switches
# 注意：动态 ablation 版模型仍然默认会构造
# morgan / exp / meth / ban(exp,morgan)
# 这里建议保持为 True / True / True
# =========================
_C.mod = CN()
_C.mod.use_morgan = True
_C.mod.use_espf = False
_C.mod.use_pubchem = False

_C.mod.use_exp = True
_C.mod.use_mut = False
_C.mod.use_meth = True
_C.mod.use_path = False

# =========================
# Encoder config
# =========================
_C.enc = CN()
_C.enc.depth = 4
_C.enc.mlp_ratio = 4
_C.enc.dropout = 0.2

# =========================
# BAN fusion config
# =========================
_C.ban = CN()
_C.ban.num_heads = 4
_C.ban.dropout = 0.1

# 这两个字段你原配置里有，先保留
_C.ban.gate_init = 1.0
_C.ban.drop_mode = "zero"

# =========================
# Transformer config
# 动态 ablation 版模型会读取 ablation_mode
# =========================
_C.trans = CN()
_C.trans.d_model = 128
_C.trans.nhead = 4
_C.trans.num_layers = 2
_C.trans.dim_feedforward = 512
_C.trans.dropout = 0.1
_C.trans.pool = "mean"

# protected token dropout
_C.trans.token_dropout = 0.2

# 默认完整模型；跑消融时由脚本覆盖
_C.trans.ablation_mode = "full"

# =========================
# Prediction head MLP
# =========================
_C.mlp = CN()
_C.mlp.mlp_in_dim = 128
_C.mlp.mlp_hidden_dim = [512, 128]


def get_cfg_defaults():
    return _C.clone()
