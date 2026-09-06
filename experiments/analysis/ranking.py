#!/usr/bin/env python
"""The two prioritisation modes a drug response model is asked to serve.

Mode 1, choosing drugs for a cell line.  Rank compounds within one held-out cell
line and take the top k.  Because the compound term dominates the response
matrix, a predictor that returns each drug's training mean already ranks
compounds well, and it makes the same recommendation for every cell line.

Mode 2, choosing cell lines for a drug.  Rank cell lines within one compound.
The per-drug-mean predictor is constant within a drug, so its ranking is
undefined.  Only a model with cell-specific output can attempt this.

The two modes come apart, and this script measures both on identical held-out
pairs so that the difference is not an artefact of different evaluation sets.

Run from the repository root after baselines.py.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
IC50 = os.path.join(ROOT, "data", "GDSC2_IC50.csv")
DRUGINFO = os.path.join(ROOT, "data", "drug_info.csv")
CASES = {
    "URINARY_TRACT": os.path.join(ROOT, "results", "cases", "urinary_predictions.csv"),
    "BREAST": os.path.join(ROOT, "results", "cases", "breast_predictions.csv"),
}
KS = (1, 3, 5, 10)
MIN_DRUGS = 20
MIN_CELLS = 8
BOOT = 10000
SEED = 20260828


def load_matrix():
    m = pd.read_csv(IC50).rename(columns={"Unnamed: 0": "depmap_id"}).set_index("depmap_id")
    info = pd.read_csv(DRUGINFO)
    name = {str(p): n for p, n in zip(info.pubchem_id, info.drug_name)}
    m.columns = [name.get(str(c), str(c)) for c in m.columns]
    return m


def ndcg_at_k(true_vals, score, k):
    """Sensitivity is low ln(IC50); gain is the negated, min-shifted response."""
    order = np.argsort(score)
    gain = -np.asarray(true_vals, float)
    gain = gain - gain.min()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gain[order][:k] * disc).sum()
    idcg = (np.sort(gain)[::-1][:k] * disc).sum()
    return dcg / idcg if idcg > 0 else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    m = load_matrix()
    per_cell, per_drug, summary = [], [], []

    for tissue, path in CASES.items():
        lg = pd.read_csv(path)
        held = set(lg.depmap_id)
        train = [i for i in m.index if i not in held]
        dmean = m.loc[train].mean(axis=0)
        lg["drug_mean"] = lg.drug_name.map(dmean)
        lg = lg.dropna(subset=["drug_mean", "true_logIC50", "pred_logIC50"])

        # ---- Mode 1: rank drugs within a cell line -------------------------
        for cell, g in lg.groupby("depmap_id"):
            if g.drug_name.nunique() < MIN_DRUGS:
                continue
            row = dict(tissue=tissue, depmap_id=cell,
                       cell_line_name=g.cell_line_name.iloc[0], n_drugs=len(g))
            tv = g.true_logIC50.to_numpy(float)
            for k in KS:
                best = set(g.nsmallest(k, "true_logIC50").drug_name)
                row[f"prec@{k}_bcgdrp"] = len(best & set(g.nsmallest(k, "pred_logIC50").drug_name)) / k
                row[f"prec@{k}_drugmean"] = len(best & set(g.nsmallest(k, "drug_mean").drug_name)) / k
                rand = np.array([len(best & set(g.iloc[rng.permutation(len(g))[:k]].drug_name)) / k
                                 for _ in range(200)]).mean()
                row[f"prec@{k}_random"] = rand
                row[f"ndcg@{k}_bcgdrp"] = ndcg_at_k(tv, g.pred_logIC50.to_numpy(float), k)
                row[f"ndcg@{k}_drugmean"] = ndcg_at_k(tv, g.drug_mean.to_numpy(float), k)
            per_cell.append(row)

        # ---- Mode 2: rank cell lines within a drug --------------------------
        for drug, g in lg.groupby("drug_name"):
            if g.depmap_id.nunique() < MIN_CELLS:
                continue
            r = stats.spearmanr(g.true_logIC50, g.pred_logIC50)[0]
            if not np.isnan(r):
                per_drug.append(dict(tissue=tissue, drug_name=drug, n_cells=g.depmap_id.nunique(),
                                     scc_bcgdrp=r, scc_drugmean=np.nan,
                                     drugmean_reason="constant within a drug, rank undefined"))

    pc = pd.DataFrame(per_cell)
    pd_ = pd.DataFrame(per_drug)
    pc.to_csv(os.path.join(OUT, "prioritisation_mode1_per_cell.csv"), index=False)
    pd_.to_csv(os.path.join(OUT, "prioritisation_mode2_per_drug.csv"), index=False)

    print("=" * 100)
    print("MODE 1  rank drugs within a held-out cell line")
    print("=" * 100)
    for tissue, g in pc.groupby("tissue", sort=False):
        print(f"\n  {tissue}  ({len(g)} cell lines)")
        for k in KS:
            a, b = g[f"prec@{k}_bcgdrp"], g[f"prec@{k}_drugmean"]
            d = a - b
            boot = np.array([rng.choice(d.values, len(d), replace=True).mean() for _ in range(BOOT)])
            W, p = stats.wilcoxon(a, b) if (d != 0).any() else (np.nan, 1.0)
            summary.append(dict(tissue=tissue, mode="rank drugs within a cell line", k=k,
                                n=len(g), bcgdrp=a.mean(), drug_mean=b.mean(),
                                random=g[f"prec@{k}_random"].mean(), delta=d.mean(),
                                ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                                wilcoxon_p=p))
            print(f"    precision@{k:<2d} BCGDRP {a.mean():.3f} | drug-mean {b.mean():.3f} | "
                  f"random {g[f'prec@{k}_random'].mean():.3f} | delta {d.mean():+.3f} "
                  f"[{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}] P={p:.3f}")

    print("\n" + "=" * 100)
    print("MODE 2  rank held-out cell lines within a drug")
    print("=" * 100)
    for tissue, g in pd_.groupby("tissue", sort=False):
        b = np.array([rng.choice(g.scc_bcgdrp.values, len(g), replace=True).mean()
                      for _ in range(BOOT)])
        print(f"  {tissue}: {len(g)} drugs | BCGDRP mean rho {g.scc_bcgdrp.mean():.4f} "
              f"[{np.percentile(b,2.5):.4f}, {np.percentile(b,97.5):.4f}] | "
              f"median {g.scc_bcgdrp.median():.4f} | positive for {100*(g.scc_bcgdrp>0).mean():.0f}%")
        summary.append(dict(tissue=tissue, mode="rank cell lines within a drug", k=np.nan,
                            n=len(g), bcgdrp=g.scc_bcgdrp.mean(), drug_mean=np.nan,
                            random=0.0, delta=np.nan,
                            ci_low=np.percentile(b, 2.5), ci_high=np.percentile(b, 97.5),
                            wilcoxon_p=np.nan))
    print("  the per-drug-mean predictor is constant within a drug, so its ranking is undefined here")

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "prioritisation_modes_summary.csv"), index=False)
    print(f"\nwrote three tables to {OUT}")
    return pc, pd_, s


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
