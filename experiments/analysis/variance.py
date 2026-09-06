#!/usr/bin/env python
"""How the variance of a drug response matrix is distributed, and what a model can win.

A cell-line-disjoint model is asked to predict a matrix whose variance is
dominated by which compound was given, not by which cell line received it.
Decomposing the observed matrix into a drug main effect, a cell main effect and
an interaction term makes the size of the available prize explicit:

    y[c,d] = mu + alpha[c] + beta[d] + gamma[c,d]

The per-drug-mean baseline captures beta exactly.  The cell main effect alpha is
unavailable under cell-line-disjoint evaluation, because a held-out cell line has
no training responses from which to estimate it.  Everything a cell-line-disjoint
model can win therefore lives in gamma.

The same decomposition is applied to the stored predictions, which shows how much
of each component the model reconstructs.

Run from the repository root.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
IC50 = os.path.join(ROOT, "data", "GDSC2_IC50.csv")
DRUGINFO = os.path.join(ROOT, "data", "drug_info.csv")
CASES = {
    "URINARY_TRACT": os.path.join(ROOT, "results", "cases", "urinary_predictions.csv"),
    "BREAST": os.path.join(ROOT, "results", "cases", "breast_predictions.csv"),
}


def decompose(long, value):
    """Two-way decomposition of a long table with columns depmap_id, drug_name, `value`."""
    y = long[value].to_numpy(float)
    mu = y.mean()
    cell = long.groupby("depmap_id")[value].transform("mean").to_numpy(float) - mu
    drug = long.groupby("drug_name")[value].transform("mean").to_numpy(float) - mu
    inter = y - mu - cell - drug
    tot = ((y - mu) ** 2).sum()
    return dict(n_pairs=len(y), n_cells=long.depmap_id.nunique(), n_drugs=long.drug_name.nunique(),
                total_ss=tot,
                drug_pct=100 * (drug ** 2).sum() / tot,
                cell_pct=100 * (cell ** 2).sum() / tot,
                interaction_pct=100 * (inter ** 2).sum() / tot)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(IC50).rename(columns={"Unnamed: 0": "depmap_id"}).set_index("depmap_id")
    info = pd.read_csv(DRUGINFO)
    name = {str(p): n for p, n in zip(info.pubchem_id, info.drug_name)}
    m.columns = [name.get(str(c), str(c)) for c in m.columns]
    full = (m.stack(dropna=True).rename("y").reset_index()
             .rename(columns={"level_1": "drug_name"}))

    rows = [dict(setting="Full GDSC2 matrix", quantity="observed response", **decompose(full, "y"))]

    for tissue, path in CASES.items():
        lg = pd.read_csv(path)
        rows.append(dict(setting=f"{tissue} hold-out", quantity="observed response",
                         **decompose(lg.rename(columns={"true_logIC50": "y"}), "y")))
        rows.append(dict(setting=f"{tissue} hold-out", quantity="BCGDRP prediction",
                         **decompose(lg.rename(columns={"pred_logIC50": "y"}), "y")))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "variance_decomposition.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=" * 96)
    print("VARIANCE DECOMPOSITION  y[c,d] = mu + cell[c] + drug[d] + interaction[c,d]")
    print("=" * 96)
    show = df[["setting", "quantity", "n_pairs", "n_cells", "n_drugs",
               "drug_pct", "cell_pct", "interaction_pct"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    f = df.iloc[0]
    print(f"\n  On the full matrix, {f.drug_pct:.1f}% of the variance is the compound, "
          f"{f.cell_pct:.1f}% is the cell line")
    print(f"  and {f.interaction_pct:.1f}% is the drug-by-cell interaction.")
    print("  The per-drug-mean baseline captures the compound term exactly.")
    print("  The cell term is unavailable under cell-line-disjoint evaluation, because a held-out")
    print("  cell line contributes no training response from which to estimate it.")
    print(f"  A cell-line-disjoint model therefore competes for the {f.interaction_pct:.1f}% "
          "interaction term alone.")
    return df


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
