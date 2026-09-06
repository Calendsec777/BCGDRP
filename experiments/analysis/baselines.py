#!/usr/bin/env python
"""DrEval-style naive predictors for the BCGDRP blind-cell benchmark.

Bernett et al., Nature Communications 17:4238 (2026) require every drug response
prediction model to be reported against predictors built from response statistics
alone.  Three are defined there:

    NaivePredictor              global training mean
    NaiveDrugMeanPredictor      per-drug training mean
    NaiveCellLineMeanPredictor  per-cell-line training mean

Under cell-line-disjoint (LCO) evaluation the test cell lines have no training
responses, so NaiveCellLineMeanPredictor degenerates to NaivePredictor; the
per-drug mean is the informative one.  This script evaluates all three on

    (a) the frozen 5-fold cell-disjoint split used for model development,
    (b) the URINARY_TRACT and BREAST disease-focused hold-outs, alongside the
        stored BCGDRP predictions, including within-cell-line ranking, and
    (c) the 16 GDSC2-unmeasured 5637 candidates that produced the experimental
        nomination.

Run from the repository root.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
IC50 = os.path.join(ROOT, "data", "GDSC2_IC50.csv")
SPLIT = os.path.join(ROOT, "data", "splits", "fivefold_seed42.json")
DRUGINFO = os.path.join(ROOT, "data", "drug_info.csv")
CASES = {
    "URINARY_TRACT": os.path.join(ROOT, "results", "cases", "urinary_predictions.csv"),
    "BREAST": os.path.join(ROOT, "results", "cases", "breast_predictions.csv"),
}
BOOT = 10000
SEED = 20260828


def load_matrix():
    m = pd.read_csv(IC50).rename(columns={"Unnamed: 0": "depmap_id"}).set_index("depmap_id")
    info = pd.read_csv(DRUGINFO)
    name = {str(p): n for p, n in zip(info.pubchem_id, info.drug_name)}
    m.columns = [name.get(str(c), str(c)) for c in m.columns]
    return m


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return dict(
        n=int(len(y)),
        RMSE=float(np.sqrt(((y - p) ** 2).mean())),
        MAE=float(np.abs(y - p).mean()),
        R2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        PCC=float(stats.pearsonr(y, p)[0]),
        SCC=float(stats.spearmanr(y, p)[0]),
    )


def naive_predictions(train, test_frame):
    """test_frame: long table with columns depmap_id, drug_name, true_logIC50."""
    drug_mean = train.mean(axis=0)
    global_mean = float(np.nanmean(train.values))
    out = test_frame.copy()
    out["naive_global"] = global_mean
    out["naive_drug_mean"] = out.drug_name.map(drug_mean)
    return out.dropna(subset=["naive_drug_mean", "true_logIC50"])


def part_a_primary(m):
    """Naive predictors on the PRIMARY split behind the main comparison table.

    Recovered from the training pipeline on 28 August 2026.  The split is produced by
    sorting the response-matrix index, indexing cell lines by their position in that
    sorted list, and applying two nested calls with a fixed seed:

        train, temp = train_test_split(cell_indices, test_size=0.2, random_state=42)
        val,  test  = train_test_split(temp,         test_size=0.5, random_state=42)

    giving 428 training, 54 validation and 54 test cell lines.  The index set reproduces
    bit-for-bit on the training machine (sklearn 1.6.1 / numpy 1.26.4) and locally
    (sklearn 1.7.0 / numpy 2.3.1), md5 of the sorted test indices 803a1c26...
    """
    from sklearn.model_selection import train_test_split
    ids = sorted(m.index.tolist())
    have = [i for i, cid in enumerate(ids) if m.loc[cid].notna().any()]
    assert have == list(range(len(ids))), "a cell line carries no response; index set differs"
    ci = np.array(sorted(have))
    tr_i, tmp_i = train_test_split(ci, test_size=0.2, random_state=42)
    va_i, te_i = train_test_split(tmp_i, test_size=0.5, random_state=42)
    tr = [ids[i] for i in tr_i]
    va = [ids[i] for i in va_i]
    te = [ids[i] for i in te_i]
    assert not (set(tr) & set(te)) and not (set(va) & set(te)), "split not cell-disjoint"

    long = (m.loc[te].stack(dropna=True).rename("true_logIC50")
            .reset_index().rename(columns={"level_1": "drug_name"}))
    lf = naive_predictions(m.loc[tr], long)
    rows = []
    for label, col in [("NaivePredictor", "naive_global"),
                       ("NaiveDrugMeanPredictor", "naive_drug_mean")]:
        rows.append(dict(predictor=label, n_train_cells=len(tr), n_val_cells=len(va),
                         n_test_cells=len(te), **metrics(lf.true_logIC50, lf[col])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "naive_baselines_primary_split.csv"), index=False)
    pd.DataFrame({"depmap_id": sorted(te)}).to_csv(
        os.path.join(OUT, "primary_split_test_cell_lines.csv"), index=False)
    print("=" * 92)
    print(f"(a0) PRIMARY SPLIT  train/val/test cell lines = {len(tr)}/{len(va)}/{len(te)}, "
          f"{len(lf)} annotated test pairs")
    print(df.round(4).to_string(index=False))
    print("\n     published values on this split (Table 1):")
    for name, r, p, sc, r2 in [("tCNNs", 1.7133, 0.7875, 0.7359, 0.6158),
                               ("DeepCDR", 1.4729, 0.8521, 0.8146, 0.7159),
                               ("DeepTTA", 1.4218, 0.8580, 0.8235, 0.7369),
                               ("GADRP", 1.4239, 0.8586, 0.8232, 0.6435),
                               ("BANDRP", 1.3771, 0.8676, 0.8358, 0.7523),
                               ("BCGDRP", 1.3644, 0.8706, 0.8381, 0.7568)]:
        dmr = df[df.predictor == "NaiveDrugMeanPredictor"].iloc[0]
        flag = "  <-- LOSES to the per-drug mean" if r > dmr.RMSE else ""
        print(f"       {name:8s} RMSE {r:.4f}  PCC {p:.4f}  SCC {sc:.4f}  R2 {r2:.4f}{flag}")
    dmr = df[df.predictor == "NaiveDrugMeanPredictor"].iloc[0]
    print(f"\n     BCGDRP vs per-drug mean: RMSE {100*(dmr.RMSE-1.3644)/dmr.RMSE:+.2f}% ; "
          f"PCC {0.8706-dmr.PCC:+.4f} ; R2 {0.7568-dmr.R2:+.4f}")
    print(f"     BCGDRP-BANDRP gap as a share of the model-baseline gap: "
          f"{100*(1.3771-1.3644)/(dmr.RMSE-1.3644):.1f}%")
    return df


def part_a_folds(m):
    """Naive predictors on the frozen 5-fold cell-disjoint split."""
    folds = json.load(open(SPLIT))["folds"]
    rows = []
    for f in folds:
        tr = [i for i in f["train_depmap_ids"] if i in m.index]
        te = [i for i in f["test_depmap_ids"] if i in m.index]
        assert not set(tr) & set(te), "train/test cell overlap"
        long = (m.loc[te].stack(dropna=True).rename("true_logIC50")
                .reset_index().rename(columns={"level_1": "drug_name"}))
        lf = naive_predictions(m.loc[tr], long)
        for label, col in [("NaivePredictor", "naive_global"),
                           ("NaiveDrugMeanPredictor", "naive_drug_mean")]:
            rows.append(dict(fold=f["fold"], predictor=label,
                             n_test_cells=len(te), **metrics(lf.true_logIC50, lf[col])))
    df = pd.DataFrame(rows)
    agg = (df.groupby("predictor")[["RMSE", "MAE", "R2", "PCC", "SCC"]]
             .agg(["mean", "std"]).round(4))
    df.to_csv(os.path.join(OUT, "naive_baselines_blind_cell_folds.csv"), index=False)
    print("=" * 92)
    print("(a) NAIVE PREDICTORS on the frozen 5-fold cell-disjoint split")
    print(df.round(4).to_string(index=False))
    print("\nmean +- sd across folds:")
    print(agg.to_string())
    return agg


def part_b_cases(m):
    """Naive vs stored BCGDRP predictions on the two disease-focused hold-outs."""
    rng = np.random.default_rng(SEED)
    summary, percell = [], []
    for tissue, path in CASES.items():
        lg = pd.read_csv(path)
        held = set(lg.depmap_id)
        tr = [i for i in m.index if i not in held]
        lf = naive_predictions(m.loc[tr], lg)
        for label, col in [("BCGDRP", "pred_logIC50"),
                           ("NaivePredictor", "naive_global"),
                           ("NaiveDrugMeanPredictor", "naive_drug_mean")]:
            summary.append(dict(tissue=tissue, predictor=label,
                                n_cells=lf.depmap_id.nunique(),
                                **metrics(lf.true_logIC50, lf[col])))
        # within-cell-line ranking: does the model resolve drugs inside one cell?
        rec = []
        for cell, g in lf.groupby("depmap_id"):
            if g.drug_name.nunique() >= 10:
                rec.append(dict(
                    tissue=tissue, depmap_id=cell,
                    cell_line_name=g.cell_line_name.iloc[0], n_drugs=g.drug_name.nunique(),
                    scc_bcgdrp=stats.spearmanr(g.true_logIC50, g.pred_logIC50)[0],
                    scc_drug_mean=stats.spearmanr(g.true_logIC50, g.naive_drug_mean)[0]))
        rec = pd.DataFrame(rec)
        rec["delta"] = rec.scc_bcgdrp - rec.scc_drug_mean
        percell.append(rec)
        W, p = stats.wilcoxon(rec.scc_bcgdrp, rec.scc_drug_mean)
        boot = np.array([rng.choice(rec.delta.values, len(rec), replace=True).mean()
                         for _ in range(BOOT)])
        print("\n" + "=" * 92)
        print(f"(b) {tissue}: within-cell-line Spearman over {len(rec)} held-out cell lines")
        print(f"    BCGDRP                 mean={rec.scc_bcgdrp.mean():.4f} "
              f"median={rec.scc_bcgdrp.median():.4f}")
        print(f"    NaiveDrugMeanPredictor mean={rec.scc_drug_mean.mean():.4f} "
              f"median={rec.scc_drug_mean.median():.4f}")
        print(f"    mean delta={rec.delta.mean():+.4f}  95% CI "
              f"[{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
              f"BCGDRP higher in {(rec.delta>0).sum()}/{len(rec)}  Wilcoxon W={W:.1f} P={p:.4g}")
        # within-drug ranking across held-out cells: naive drug mean is constant -> undefined
        rd = []
        for drug, g in lf.groupby("drug_name"):
            if g.depmap_id.nunique() >= 8:
                r = stats.spearmanr(g.true_logIC50, g.pred_logIC50)[0]
                if not np.isnan(r):
                    rd.append(dict(tissue=tissue, drug_name=drug,
                                   n_cells=g.depmap_id.nunique(), scc_bcgdrp=r))
        rd = pd.DataFrame(rd)
        rd.to_csv(os.path.join(OUT, f"within_drug_scc_{tissue.lower()}.csv"), index=False)
        print(f"    within-DRUG Spearman across held-out cells ({len(rd)} drugs, >=8 cells): "
              f"mean={rd.scc_bcgdrp.mean():.4f} median={rd.scc_bcgdrp.median():.4f} "
              f"frac>0={(rd.scc_bcgdrp>0).mean():.2f}  "
              f"[NaiveDrugMeanPredictor is constant within a drug -> rho undefined]")
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "naive_vs_bcgdrp_disease_holdouts.csv"), index=False)
    pd.concat(percell).to_csv(os.path.join(OUT, "within_cell_scc_disease_holdouts.csv"), index=False)
    print("\n" + "=" * 92)
    print("(b) pooled metrics on annotated held-out pairs")
    print(s.round(4).to_string(index=False))
    return s


def part_c_nomination(m):
    """Did the model add information over a no-model ranking of the 16 candidates?

    5637 is a held-out URINARY_TRACT cell line.  Sixteen drugs in the panel have no
    GDSC2 response for it; Daporinad was nominated from that set and assayed.  Two
    ranking rules that use no cell-line features at all are compared with the model:
    the drug's mean response over the model-visible (non-URINARY_TRACT) cell lines,
    and its mean response over the other held-out urinary tract cell lines.
    """
    lg = pd.read_csv(CASES["URINARY_TRACT"])
    urinary = set(lg.depmap_id)
    id5637 = lg.loc[lg.cell_line_name == "5637", "depmap_id"].iloc[0]
    measured = set(lg.loc[lg.cell_line_name == "5637", "drug_name"])
    cand = [d for d in m.columns if d not in measured]
    visible = [i for i in m.index if i not in urinary]
    other_ur = [i for i in urinary if i != id5637]

    rows = []
    for d in cand:
        col = m[d]
        rows.append(dict(
            drug=d,
            mean_lnIC50_visible_cells=col.loc[visible].mean(),
            n_visible=int(col.loc[visible].notna().sum()),
            mean_lnIC50_other_urinary=col.loc[other_ur].mean(),
            n_other_urinary=int(col.loc[other_ur].notna().sum())))
    b = pd.DataFrame(rows)
    b["rank_visible_cell_mean"] = b.mean_lnIC50_visible_cells.rank().astype(int)
    b["rank_other_urinary_mean"] = b.mean_lnIC50_other_urinary.rank().astype(int)
    b = b.sort_values("mean_lnIC50_visible_cells")
    b.to_csv(os.path.join(OUT, "nomination_baseline_audit_5637.csv"), index=False)
    print("\n" + "=" * 92)
    print(f"(c) NO-MODEL RANKING of the {len(cand)} GDSC2-unmeasured 5637 candidates")
    print("    (ascending mean observed ln(IC50); most potent first)")
    print(b.round(3).to_string(index=False))
    r = b.set_index("drug")
    print(f"\n    Daporinad by drug mean over model-visible cells : "
          f"rank {r.rank_visible_cell_mean['Daporinad']}/{len(b)}")
    print(f"    Daporinad by drug mean over other urinary lines : "
          f"rank {r.rank_other_urinary_mean['Daporinad']}/{len(b)} "
          f"(observed in {r.n_other_urinary['Daporinad']} of them, "
          f"mean ln IC50 = {r.mean_lnIC50_other_urinary['Daporinad']:.3f} -> "
          f"{np.exp(r.mean_lnIC50_other_urinary['Daporinad'])*1000:.1f} nM)")
    print("    BCGDRP rank reported in the manuscript              : 2/16")
    return b


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    m = load_matrix()
    print(f"GDSC2 response matrix: {m.shape[0]} cell lines x {m.shape[1]} drugs, "
          f"{int(m.notna().sum().sum())} observed pairs\n")
    part_a_primary(m)
    part_a_folds(m)
    part_b_cases(m)
    part_c_nomination(m)
    print(f"\nwrote results to {OUT}")
