from pathlib import Path

from yacs.config import CfgNode as CN

_C = CN()

# =========================
# Path config
# =========================
_C.path = CN()
ROOT = Path(__file__).resolve().parents[2]

_C.path.savedir = str(ROOT / "outputs" / "fusion")

_C.path.response = str(ROOT / "data" / "GDSC2_IC50.csv")
_C.path.methylation = str(ROOT / "data" / "cell" / "geo_methylation_cosmic.csv")
_C.path.expression = str(ROOT / "data" / "cell" / "geo_expression_cosmic.csv")

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
# Keep the three original raw tokens: drug morgan + exp + meth
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
# Fusion config
# =========================
_C.fusion = CN()
# type: ban / concat_mlp / cross_attn / none
_C.fusion.type = "ban"
# pair: exp_drug / meth_drug (ignored when type == 'none')
_C.fusion.pair = "exp_drug"

# BAN settings
_C.fusion.ban_num_heads = 4
_C.fusion.ban_dropout = 0.1

# Concat+MLP settings
_C.fusion.concat_hidden_ratio = 2
_C.fusion.concat_dropout = 0.1

# Cross-attention settings
_C.fusion.cross_num_heads = 4
_C.fusion.cross_dropout = 0.1
_C.fusion.cross_ff_ratio = 2

# none baseline: zero placeholder token, still keeps 4 input slots to transformer
_C.fusion.none_mode = "zero"

# =========================
# Transformer config
# =========================
_C.trans = CN()
_C.trans.d_model = 128
_C.trans.nhead = 4
_C.trans.num_layers = 2
_C.trans.dim_feedforward = 512
_C.trans.dropout = 0.1
_C.trans.pool = "mean"
_C.trans.token_dropout = 0.2

# =========================
# Prediction head MLP
# =========================
_C.mlp = CN()
_C.mlp.mlp_in_dim = 128
_C.mlp.mlp_hidden_dim = [512, 128]


def get_cfg_defaults():
    return _C.clone()
