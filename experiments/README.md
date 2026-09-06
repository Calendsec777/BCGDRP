# Experiments

`analysis/` contains the statistical, calibration, ranking, modality and dose-response analyses used in
the manuscript. `ablations/` contains consolidated module and fusion ablation runners. Released
outputs are under `results/`; values intended for direct figure and table reuse are under
`source_data/`.

Most scripts can be run from the repository root. `maps.py` and `modalities.py` additionally require
authorized local molecular matrices described in `DATA_ACCESS.md`; their released numerical outputs
are already available under `results/` and `source_data/`.
