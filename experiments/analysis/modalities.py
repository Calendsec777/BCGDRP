#!/usr/bin/env python
"""How much response information each molecular modality carries, before any architecture.

The modality ablations in the manuscript remove an input from BCGDRP and read the
change in error.  That measures the modality and the architecture together, so a
modality can look uninformative because the encoder fails to use it.  This script
separates the two by asking, for each compound independently, how well a ridge
regression on one modality alone predicts response across cell lines, fitted on the
training cell lines of the primary split and evaluated on the held-out ones.

The task is the within-drug ranking of cell lines, which is the one a per-drug-mean
predictor cannot attempt.  The purpose here is the comparison BETWEEN modalities on a
common task and a common split: how much response information each molecular layer
carries, independently of any particular architecture.

Also reported: where the held-out cell lines sit relative to training cells in each
feature space, and whether distance from the training set predicts error.

Run from the repository root.
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
DATA = os.path.join(ROOT, "data")
CELL = os.path.join(DATA, "cell")
MIN_TRAIN, MIN_TEST = 60, 8
ALPHAS = np.logspace(0, 5, 12)
SEED = 20260828

MODALITIES = [("expression", "geo_expression_cosmic.csv"),
              ("methylation", "geo_methylation_cosmic.csv"),
              ("mutation", "geo_mutation_cosmic.csv"),
              ("pathway", "pathway_cosmic.csv")]


def response():
    m = pd.read_csv(os.path.join(DATA, "GDSC2_IC50.csv")).rename(
        columns={"Unnamed: 0": "depmap_id"}).set_index("depmap_id")
    info = pd.read_csv(os.path.join(DATA, "drug_info.csv"))
    name = {str(p): n for p, n in zip(info.pubchem_id, info.drug_name)}
    m.columns = [name.get(str(c), str(c)) for c in m.columns]
    return m


def primary_split(index):
    ids = sorted(index.tolist())
    ci = np.arange(len(ids))
    tr, tmp = train_test_split(ci, test_size=0.2, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    return ([ids[i] for i in tr], [ids[i] for i in va], [ids[i] for i in te])


def load_modalities():
    out = {}
    for name, fn in MODALITIES:
        d = pd.read_csv(os.path.join(CELL, fn), index_col=0)
        d = d.loc[:, d.std(axis=0) > 0]
        out[name] = d
    return out


def per_drug_ridge(resp, mods, train, test):
    """Within-drug ranking accuracy from each modality alone."""
    rows = []
    scalers = {}
    Xtr, Xte = {}, {}
    for name, d in mods.items():
        tr = [c for c in train if c in d.index]
        te = [c for c in test if c in d.index]
        sc = StandardScaler().fit(d.loc[tr].values)
        scalers[name] = (sc, tr, te)
        Xtr[name] = sc.transform(d.loc[tr].values)
        Xte[name] = sc.transform(d.loc[te].values)
    combos = list(mods) + ["expression+methylation", "all four"]
    for drug in resp.columns:
        y = resp[drug].dropna()
        for combo in combos:
            parts = (["expression", "methylation"] if combo == "expression+methylation"
                     else list(mods) if combo == "all four" else [combo])
            trc = [c for c in scalers[parts[0]][1] if c in y.index]
            tec = [c for c in scalers[parts[0]][2] if c in y.index]
            for pp in parts[1:]:
                trc = [c for c in trc if c in scalers[pp][1]]
                tec = [c for c in tec if c in scalers[pp][2]]
            if len(trc) < MIN_TRAIN or len(tec) < MIN_TEST:
                continue
            def stack(cells, which):
                return np.hstack([mods[pp].loc[cells].values for pp in parts]) if len(parts) > 1 \
                    else mods[parts[0]].loc[cells].values
            A, B = stack(trc, 0), stack(tec, 1)
            sc = StandardScaler().fit(A)
            model = RidgeCV(alphas=ALPHAS).fit(sc.transform(A), y.loc[trc].values)
            pred = model.predict(sc.transform(B))
            obs = y.loc[tec].values
            if np.std(pred) == 0:
                continue
            rows.append(dict(drug_name=drug, modality=combo, n_train=len(trc), n_test=len(tec),
                             spearman=stats.spearmanr(obs, pred)[0],
                             pearson=stats.pearsonr(obs, pred)[0],
                             alpha=float(model.alpha_)))
    return pd.DataFrame(rows).dropna(subset=["spearman"])


def feature_space(mods, train, test):
    """Where held-out cells sit relative to training cells, and how far away they are."""
    rows, coords = [], {}
    for name, d in mods.items():
        tr = [c for c in train if c in d.index]
        te = [c for c in test if c in d.index]
        sc = StandardScaler().fit(d.loc[tr].values)
        A, B = sc.transform(d.loc[tr].values), sc.transform(d.loc[te].values)
        p = PCA(n_components=2, random_state=SEED).fit(A)
        coords[name] = (pd.DataFrame(p.transform(A), index=tr, columns=["pc1", "pc2"]),
                        pd.DataFrame(p.transform(B), index=te, columns=["pc1", "pc2"]),
                        p.explained_variance_ratio_)
        # distance from each held-out cell to its nearest training cell, in the full space
        d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        rows.append(pd.DataFrame(dict(depmap_id=te, modality=name,
                                      nn_distance=np.sqrt(d2.min(1)),
                                      mean_distance=np.sqrt(d2.mean(1)))))
    return pd.concat(rows, ignore_index=True), coords


def main():
    os.makedirs(OUT, exist_ok=True)
    resp = response()
    mods = load_modalities()
    train, val, test = primary_split(resp.index)
    print("=" * 100)
    print(f"MODALITY INFORMATION  primary split, {len(train)} training and {len(test)} held-out cell lines")
    print("=" * 100)
    for name, d in mods.items():
        print(f"  {name:22s} {d.shape[1]:5d} features, {d.shape[0]} cell lines profiled")

    pd_ = per_drug_ridge(resp, mods, train, test)
    pd_.to_csv(os.path.join(OUT, "modality_within_drug_ridge.csv"), index=False)
    print(f"\n  within-drug Spearman from each modality alone, ridge on the training cells")
    summ = (pd_.groupby("modality")
              .agg(n_drugs=("drug_name", "nunique"), mean=("spearman", "mean"),
                   median=("spearman", "median"), frac_pos=("spearman", lambda s: (s > 0).mean()))
              .sort_values("mean", ascending=False))
    print(summ.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  These are relative comparisons between modalities on a common task and a common"
          "\n  split. They measure the information each molecular layer carries about response,"
          "\n  independently of any particular architecture.")

    # paired comparison of the two cellular modalities
    w = pd_.pivot_table(index="drug_name", columns="modality", values="spearman")
    both = w[["expression", "methylation"]].dropna()
    W, p = stats.wilcoxon(both.expression, both.methylation)
    print(f"\n  expression against methylation, paired over {len(both)} compounds: "
          f"mean {both.expression.mean():.4f} vs {both.methylation.mean():.4f}, "
          f"expression higher for {int((both.expression > both.methylation).sum())}/{len(both)}, "
          f"Wilcoxon P = {p:.2g}")
    if "expression+methylation" in w.columns:
        tri = w[["expression", "methylation", "expression+methylation"]].dropna()
        gain = tri["expression+methylation"] - tri[["expression", "methylation"]].max(axis=1)
        print(f"  adding methylation to expression: mean change {gain.mean():+.4f}, "
              f"positive for {int((gain > 0).sum())}/{len(tri)} compounds")

    dist, coords = feature_space(mods, train, test)
    dist.to_csv(os.path.join(OUT, "modality_feature_distance.csv"), index=False)
    for name, (A, B, ev) in coords.items():
        A.assign(split="train", modality=name).to_csv(
            os.path.join(OUT, f"modality_pca_{name}_train.csv"))
        B.assign(split="held-out", modality=name).to_csv(
            os.path.join(OUT, f"modality_pca_{name}_test.csv"))
        print(f"  PCA {name:14s} first two components explain "
              f"{100*ev.sum():.1f}% of variance")

    # does distance from the training set predict error?
    diag = pd.read_csv(os.path.join(ROOT, "source_data",
                                    "Fig2_case_prediction_diagnostics.csv"))
    err = diag.groupby("depmap_id").residual.apply(lambda r: np.sqrt((r ** 2).mean())).rename("rmse")
    rows = []
    for name in mods:
        d = dist[dist.modality == name].set_index("depmap_id").join(err, how="inner").dropna()
        if len(d) >= 8:
            r, p = stats.spearmanr(d.nn_distance, d.rmse)
            rows.append(dict(modality=name, n_cells=len(d), spearman=r, p_value=p))
    if rows:
        e = pd.DataFrame(rows)
        e.to_csv(os.path.join(OUT, "modality_distance_vs_error.csv"), index=False)
        print("\n  distance from the training set against per-cell RMSE "
              "(disease-focused hold-outs):")
        print(e.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote modality tables to {OUT}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()


def holdout_distances():
    """Distance to the training set computed within each disease-focused hold-out.

    The primary-split distances above are indexed by that split's held-out cells,
    which barely overlap the tissue hold-outs, so relating distance to per-cell error
    needs the distance recomputed against each hold-out's own training set.
    """
    resp = response()
    mods = load_modalities()
    cases = {
        "URINARY_TRACT": os.path.join(ROOT, "results", "cases", "urinary_predictions.csv"),
        "BREAST": os.path.join(ROOT, "results", "cases", "breast_predictions.csv"),
    }
    rows, summary = [], []
    for tissue, path in cases.items():
        lg = pd.read_csv(path)
        held = set(lg.depmap_id)
        err = (lg.assign(res=lg.pred_logIC50 - lg.true_logIC50)
                 .groupby("depmap_id").res.apply(lambda r: np.sqrt((r ** 2).mean())))
        for name, d in mods.items():
            tr = [c for c in resp.index if c not in held and c in d.index]
            te = [c for c in sorted(held) if c in d.index]
            if len(te) < 8:
                continue
            sc = StandardScaler().fit(d.loc[tr].values)
            A, B = sc.transform(d.loc[tr].values), sc.transform(d.loc[te].values)
            nn = np.sqrt((((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)).min(1))
            f = pd.DataFrame(dict(tissue=tissue, modality=name, depmap_id=te, nn_distance=nn))
            f["rmse"] = f.depmap_id.map(err)
            f = f.dropna(subset=["rmse"])
            rows.append(f)
            r, p = stats.spearmanr(f.nn_distance, f.rmse)
            summary.append(dict(tissue=tissue, modality=name, n_cells=len(f),
                                spearman=r, p_value=p))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "modality_holdout_distance_vs_error.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "modality_holdout_distance_summary.csv"), index=False)
    print("\n" + "=" * 100)
    print("DISTANCE TO THE TRAINING SET AGAINST PER-CELL ERROR, within each hold-out")
    print("=" * 100)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out, s
