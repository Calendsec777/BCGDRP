"""Load the two configurations used for the released experiments."""

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class Config(dict):
    """Small nested mapping with both item and attribute access."""

    def __init__(self, values=None, **kwargs):
        super().__init__()
        for key, value in dict(values or {}, **kwargs).items():
            self[key] = value

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = __setitem__

    def clone(self):
        return deepcopy(self)


def load_config(model: str = "bcgdrp", path: str | Path | None = None) -> Config:
    """Load a YAML configuration without embedding machine-specific paths."""
    if path is None:
        filename = "bcgdrp.yaml" if model == "bcgdrp" else "baseline.yaml"
        path = CONFIG_DIR / filename
    with Path(path).open("r", encoding="utf-8") as handle:
        return Config(yaml.safe_load(handle))


def configure_paths(
    cfg: Config,
    data_dir: str | Path = ROOT / "data",
    output_dir: str | Path = ROOT / "outputs",
) -> Config:
    """Attach local input/output paths to a cloned experiment configuration."""
    cfg = cfg.clone()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cfg.path = Config()
    cfg.path.response = str(data_dir / "GDSC2_IC50.csv")
    cfg.path.expression = str(data_dir / "cell" / "geo_expression_cosmic.csv")
    cfg.path.methylation = str(data_dir / "cell" / "geo_methylation_cosmic.csv")
    cfg.path.morgan = str(data_dir / "drug" / "morgan_encoding.pkl")
    cfg.path.savedir = str(output_dir)
    return cfg


def get_bcgdrp_config() -> Config:
    return load_config("bcgdrp")


def get_baseline_config() -> Config:
    return load_config("baseline")
