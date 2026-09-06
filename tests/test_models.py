"""Checkpoint compatibility tests (requires the recorded PyTorch environment)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bcgdrp.baseline import BANDRPGate
from bcgdrp.config import load_config
from bcgdrp.model import BCGDRP

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,model_class,config_name",
    [("bcgdrp", BCGDRP, "bcgdrp.yaml"), ("bandrp_gate", BANDRPGate, "baseline.yaml")],
)
def test_checkpoint_forward(name, model_class, config_name):
    cfg = load_config(path=ROOT / "configs" / config_name)
    model = model_class(714, 1, 603, 1, **cfg)
    checkpoint = ROOT / "checkpoints" / "files" / f"{name}_seed2020.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        output, _ = model(
            [torch.zeros(2, 2048)],
            [torch.zeros(2, 714), torch.zeros(2, 1), torch.zeros(2, 603), torch.zeros(2, 1)],
        )
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
