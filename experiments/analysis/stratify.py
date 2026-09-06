#!/usr/bin/env python
"""Per-stratum performance, with the strata fixed by an external annotation.

The compounds that gain most from the drug-conditioned architecture looked, on
inspection, like selective kinase inhibitors.  Reading a pattern off a ranked
list is not a test, so the strata here come from the official GDSC screened-compound
annotation (release 8.5, TARGET_PATHWAY), which was written without reference to
this model, and the grouping rule is fixed before any result is examined:

    a target pathway with at least MIN_DRUGS evaluated compounds is its own
    stratum; the remainder are pooled as "Other pathways".

Reports per-pathway, per-drug and per-cell-line breakdowns for the two
disease-focused hold-outs and for the CTRPv2 cross-screen comparison, each with
the count entering and the count skipped.

Run from the repository root, after ctrpv2_drug_level.py and prioritisation_modes.py.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
DATA = os.path.join(ROOT, "data", "ctrp")
MIN_DRUGS = 5
BOOT = 10000
SEED = 20260828


def annotation():
    a = pd.read_csv(os.path.join(DATA, "drug_target_annotation.csv"))
    return a.dropna(subset=["drug_name"]).drop_duplicates("drug_name").set_index("drug_name")


def collapse(series, min_drugs=MIN_DRUGS):
    counts = series.value_counts()
    keep = set(counts[counts >= min_drugs].index)
    return series.where(series.isin(keep), "Other pathways").fillna("Unannotated")


def main():
    rng = np.random.default_rng(SEED)
    ann = annotation()

    # ---- CTRPv2 cross-screen, the pre-specified class test ---------------------
    c = pd.read_csv(os.path.join(OUT, "ctrpv2_drug_level_seed_aggregated.csv"))
    c["pathway"] = collapse(c.drug_name.map(ann.TARGET_PATHWAY))
    print("=" * 104)
    print("PRE-SPECIFIED TARGET-PATHWAY STRATA, CTRPv2 cross-screen rank transfer")
    print("=" * 104)
    print(f"  annotated {c.drug_name.map(ann.TARGET_PATHWAY).notna().sum()}/{len(c)} evaluated compounds")
    rows = []
    for p, g in c.groupby("pathway"):
        d = g.bcgdrp - g.bandrp_gate
        b = np.array([rng.choice(d.values, len(d), replace=True).mean() for _ in range(BOOT)])
        rows.append(dict(stratum=p, n_drugs=len(g), bcgdrp=g.bcgdrp.mean(),
                         bandrp_gate=g.bandrp_gate.mean(), delta=d.mean(),
                         ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5),
                         n_won=int((d > 0).sum())))
    s = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # omnibus: does the difference depend on stratum at all?
    groups = [g.bcgdrp.values - g.bandrp_gate.values for _, g in c.groupby("pathway")]
    H, pk = stats.kruskal(*[x for x in groups if len(x) >= 3])
    print(f"\n  Kruskal-Wallis across strata: H = {H:.2f}, P = {pk:.3f}")
    print("  This is the pre-specified test. A ranked list of the largest per-drug gains is not.")
    s.to_csv(os.path.join(OUT, "stratified_ctrpv2_by_pathway.csv"), index=False)

    # ---- hold-out within-drug ranking by pathway ------------------------------
    print("\n" + "=" * 104)
    print("PER-PATHWAY WITHIN-DRUG RANKING IN THE DISEASE-FOCUSED HOLD-OUTS")
    print("=" * 104)
    pm = pd.read_csv(os.path.join(OUT, "prioritisation_mode2_per_drug.csv"))
    pm["pathway"] = collapse(pm.drug_name.map(ann.TARGET_PATHWAY))
    rows = []
    for (t, p), g in pm.groupby(["tissue", "pathway"]):
        rows.append(dict(tissue=t, stratum=p, n_drugs=len(g),
                         mean_scc=g.scc_bcgdrp.mean(), median_scc=g.scc_bcgdrp.median(),
                         frac_positive=(g.scc_bcgdrp > 0).mean()))
    h = pd.DataFrame(rows).sort_values(["tissue", "mean_scc"], ascending=[True, False])
    print(h.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    h.to_csv(os.path.join(OUT, "stratified_holdout_by_pathway.csv"), index=False)

    # ---- ceiling by pathway ---------------------------------------------------
    cp = os.path.join(OUT, "model_vs_ceiling_per_drug.csv")
    if os.path.exists(cp):
        v = pd.read_csv(cp)
        v["pathway"] = collapse(v.drug_name.map(ann.TARGET_PATHWAY))
        rows = []
        for p, g in v.groupby("pathway"):
            rows.append(dict(stratum=p, n_drugs=len(g), ceiling=g.ceiling_rho.mean(),
                             bcgdrp=g.bcgdrp.mean(),
                             pct_of_ceiling=100 * g.bcgdrp.mean() / g.ceiling_rho.mean()))
        z = pd.DataFrame(rows).sort_values("ceiling", ascending=False)
        print("\n" + "=" * 104)
        print("SCREEN-AGREEMENT CEILING AND ATTAINMENT BY PATHWAY")
        print("=" * 104)
        print(z.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        z.to_csv(os.path.join(OUT, "stratified_ceiling_by_pathway.csv"), index=False)

    # ---- per-cell-line breakdown ----------------------------------------------
    pc = pd.read_csv(os.path.join(OUT, "within_cell_scc_disease_holdouts.csv"))
    pc.to_csv(os.path.join(OUT, "stratified_per_cell_line.csv"), index=False)
    print(f"\n  per-cell-line table: {len(pc)} held-out cell lines across "
          f"{pc.tissue.nunique()} tissue hold-outs")
    print(f"wrote stratified tables to {OUT}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
