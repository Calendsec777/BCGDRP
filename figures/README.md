# Figure builders

Each script reads released values from `source_data/` or `results/` and writes publication-size PDF and
PNG artwork to `outputs/figures/`. `style.py` enforces canvas dimensions, typography and palette
checks; `panels.py` contains the shared response-matrix panel.

The static project overview displayed by GitHub is `docs/flowchart.png`. `fig1bc.py` rebuilds the
data-driven drug-space and cell-space panels of Figure 1; the other scripts rebuild the released
quantitative figures.

Run scripts from the repository root. No restricted molecular matrix is required to rebuild figures
from the committed source data.
