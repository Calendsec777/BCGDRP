# Checkpoints

`files/` contains the validation-selected BCGDRP and BANDRP-Gate weights for seeds 2020--2024. The
filenames are `<model>_seed<seed>.pt`. `manifest.csv` records byte sizes and SHA-256 digests.

Validate all ten files with:

```bash
python scripts/check.py
```

The files contain PyTorch state dictionaries only; they do not contain optimizer state, training data
or executable source code. Load them only with the matching released model definitions and PyTorch
environment.
