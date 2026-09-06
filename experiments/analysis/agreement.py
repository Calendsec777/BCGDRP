#!/usr/bin/env python
"""How well do two independent screens agree, and how close does the model get?

The manuscript reports a within-drug Spearman correlation of 0.264 against CTRPv2
for cell lines absent from the GDSC response matrix.  That number is only
interpretable against a ceiling, because CTRPv2 and GDSC differ in assay format,
exposure time, seeding density and response summary, so even a perfect model of
GDSC response would not reproduce the CTRPv2 ranking exactly.

This script measures the ceiling directly.  For the drug-cell pairs measured by
BOTH screens, it computes the same within-drug Spearman correlation between the
CTRPv2 activity area and the GDSC ln(IC50), over the same 112 shared compounds
and the same metric.  That is the agreement two real screens achieve on identical
biology, and no model scored against CTRPv2 can exceed it other than by chance.

Direction: CTRPv2 activity area rises with potency, GDSC ln(IC50) rises with
resistance, so agreement is measured between -AAC and ln(IC50).

Run from the repository root.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
PAIRS = os.path.join(ROOT, "data", "ctrp",
                     "pairs.csv")
IC50 = os.path.join(ROOT, "data", "GDSC2_IC50.csv")
DRUGINFO = os.path.join(ROOT, "data", "drug_info.csv")
MODEL = os.path.join(OUT, "ctrpv2_drug_level_seed_aggregated.csv")
MIN_CELLS = 10
BOOT = 10000
SEED = 20260828


def load_gdsc():
    m = pd.read_csv(IC50).rename(columns={"Unnamed: 0": "depmap_id"}).set_index("depmap_id")
    info = pd.read_csv(DRUGINFO)
    name = {str(p): n for p, n in zip(info.pubchem_id, info.drug_name)}
    m.columns = [name.get(str(c), str(c)) for c in m.columns]
    return (m.stack(dropna=True).rename("gdsc_lnIC50").reset_index()
             .rename(columns={"level_1": "gdsc_drug_name"}))


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ctrp = pd.read_csv(PAIRS, low_memory=False)
    ctrp = ctrp[ctrp.depmap_id.notna() & ctrp.aac.notna()]
    overlap = ctrp[ctrp.in_gdsc_response.astype(str).str.lower() == "true"].copy()
    gdsc = load_gdsc()

    j = overlap.merge(gdsc, on=["depmap_id", "gdsc_drug_name"], how="inner")
    j = j.drop_duplicates(["depmap_id", "gdsc_drug_name"])
    print("=" * 92)
    print("SCREEN-AGREEMENT CEILING: CTRPv2 against GDSC2 on pairs measured by both")
    print("=" * 92)
    print(f"  CTRPv2 records with a mapped cell line and an AAC value : {len(ctrp):,}")
    print(f"  of which the cell line also has GDSC responses          : {len(overlap):,}")
    print(f"  joined to a GDSC response for the same drug-cell pair   : {len(j):,}")
    print(f"  distinct cell lines {j.depmap_id.nunique()}, distinct drugs {j.gdsc_drug_name.nunique()}")

    rows = []
    for drug, g in j.groupby("gdsc_drug_name"):
        if g.depmap_id.nunique() >= MIN_CELLS:
            rho = stats.spearmanr(-g.aac, g.gdsc_lnIC50)[0]
            if not np.isnan(rho):
                rows.append(dict(drug_name=drug, n_cells=g.depmap_id.nunique(), ceiling_rho=rho))
    ceil = pd.DataFrame(rows).sort_values("ceiling_rho", ascending=False)
    ceil.to_csv(os.path.join(OUT, "screen_agreement_ceiling_per_drug.csv"), index=False)

    n = len(ceil)
    boot = np.array([rng.choice(ceil.ceiling_rho.values, n, replace=True).mean()
                     for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  drugs with >= {MIN_CELLS} shared cell lines: {n}")
    print(f"  CEILING, macro Spearman over drugs : {ceil.ceiling_rho.mean():.4f} "
          f"(95% CI {lo:.4f} to {hi:.4f})")
    print(f"  CEILING, median over drugs         : {ceil.ceiling_rho.median():.4f}")
    print(f"  drugs where the two screens disagree in direction: "
          f"{int((ceil.ceiling_rho < 0).sum())}/{n}")
    for thr in (30, 50):
        s = ceil[ceil.n_cells >= thr]
        print(f"    restricted to >= {thr} shared cell lines (n={len(s)}): "
              f"macro {s.ceiling_rho.mean():.4f}, median {s.ceiling_rho.median():.4f}")
    return ceil, j


if __name__ == "__main__":
    main()


def matched(ceil, model_path=MODEL):
    """Compare model and ceiling on the SAME drugs, paired."""
    rng = np.random.default_rng(SEED)
    mod = pd.read_csv(model_path)
    m = ceil.merge(mod[["drug_name", "bcgdrp", "bandrp_gate", "n_cells"]],
                   on="drug_name", how="inner", suffixes=("_ceiling", "_model"))
    m["frac_of_ceiling"] = m.bcgdrp / m.ceiling_rho
    m = m.sort_values("ceiling_rho", ascending=False)
    m.to_csv(os.path.join(OUT, "model_vs_ceiling_per_drug.csv"), index=False)

    n = len(m)
    print("\n" + "=" * 92)
    print(f"MATCHED COMPARISON on the {n} drugs present in both analyses")
    print("=" * 92)
    print(f"  ceiling  (CTRPv2 vs GDSC2, same pairs) macro rho : {m.ceiling_rho.mean():.4f}")
    print(f"  BCGDRP      (model vs CTRPv2)          macro rho : {m.bcgdrp.mean():.4f}")
    print(f"  BANDRP-gate (model vs CTRPv2)          macro rho : {m.bandrp_gate.mean():.4f}")
    ratio = m.bcgdrp.mean() / m.ceiling_rho.mean()
    boot = np.array([
        (lambda s: s.bcgdrp.mean() / s.ceiling_rho.mean())(m.iloc[rng.integers(0, n, n)])
        for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  BCGDRP reaches {100*ratio:.1f}% of the achievable ceiling "
          f"(95% CI {100*lo:.1f}% to {100*hi:.1f}%)")
    print(f"  BANDRP-gate reaches {100*m.bandrp_gate.mean()/m.ceiling_rho.mean():.1f}%")
    r, p = stats.spearmanr(m.ceiling_rho, m.bcgdrp)
    print(f"\n  corr(ceiling rho, model rho) across drugs = {r:+.3f}, P = {p:.2g}")
    print("    a positive value means the model does better on exactly the compounds whose")
    print("    response is reproducible between screens, which is the expected behaviour")
    print(f"\n  drugs where the model exceeds the ceiling: {int((m.bcgdrp > m.ceiling_rho).sum())}/{n}")
    print(f"  per-drug fraction of ceiling: median {m.frac_of_ceiling.median():.2f}, "
          f"IQR [{m.frac_of_ceiling.quantile(.25):.2f}, {m.frac_of_ceiling.quantile(.75):.2f}]")
    k = 6
    print(f"\n  most reproducible compounds between screens (top {k}):")
    print(m.head(k)[["drug_name", "n_cells_ceiling", "ceiling_rho", "bcgdrp", "frac_of_ceiling"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  least reproducible (bottom {k}):")
    print(m.tail(k)[["drug_name", "n_cells_ceiling", "ceiling_rho", "bcgdrp", "frac_of_ceiling"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return m
