#!/usr/bin/env python
"""Descriptive maps of the two input modalities and of the response matrix itself.

Run from the repository root.  These are views of data the benchmark already uses,
so they add no claim.  They exist because the paper's central quantity, that
compound identity carries 67.4% of the variance, is a number the reader has to take
on trust unless the matrix is shown banding by compound.

Joint coverage was counted before any of this was written: 169/169 benchmark
compounds carry both a Morgan fingerprint and a GDSC target pathway, 536/536 cell
lines carry a tissue label and expression rows, and the matrix holds 81,467
observed entries in 536 x 169, which is the benchmark size reported in the paper.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 20260828
D = "data"
SD = "source_data"
OUT = "results/analysis"


def load():
    ic = pd.read_csv(os.path.join(D, "GDSC2_IC50.csv"), index_col=0)
    ic.columns = [str(c) for c in ic.columns]
    mor = {str(k): np.asarray(v, dtype=np.uint8)
           for k, v in pickle.load(open(os.path.join(D, "drug/morgan_encoding.pkl"), "rb")).items()}
    tgt = pd.read_csv(os.path.join(SD, "drug_target_annotation.csv"))
    tgt["pubchem_id"] = tgt.pubchem_id.astype(str)
    tgt = tgt.drop_duplicates("pubchem_id").set_index("pubchem_id")
    cli = pd.read_csv(os.path.join(D, "cell/cell_line_info.csv")).drop_duplicates("DepMap_ID")
    cli = cli.set_index("DepMap_ID")
    return ic, mor, tgt, cli


def drug_space(ic, mor, tgt):
    """169 compounds in Morgan space, laid out by Tanimoto distance."""
    ids = [d for d in ic.columns if d in mor]
    assert len(ids) == ic.shape[1], f"{ic.shape[1] - len(ids)} compounds lack a fingerprint"
    X = np.vstack([mor[d] for d in ids])
    dist = squareform(pdist(X, metric="jaccard"))
    xy = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=18, random_state=SEED).fit_transform(dist)
    d = pd.DataFrame({"pubchem_id": ids, "x": xy[:, 0], "y": xy[:, 1]})
    d["drug_name"] = [tgt.drug_name.get(i, i) for i in ids]
    d["target_pathway"] = [tgt.TARGET_PATHWAY.get(i, "Unknown") for i in ids]
    d["n_cells"] = [int(ic[i].notna().sum()) for i in ids]
    d["response_sd"] = [float(ic[i].std()) for i in ids]
    d["response_mean"] = [float(ic[i].mean()) for i in ids]
    return d


def cell_space(ic, cli):
    """536 cell lines in expression space, with the held-out groups marked."""
    exp = pd.read_csv(os.path.join(D, "cell/geo_expression_cosmic.csv"), index_col=0)
    cells = [c for c in ic.index if c in exp.index]
    assert len(cells) == ic.shape[0], f"{ic.shape[0] - len(cells)} cell lines lack expression"
    A = exp.loc[cells].astype(np.float32).values
    xy = PCA(n_components=2, random_state=SEED).fit_transform(A - A.mean(0))
    ccle = cli.CCLE_Name.astype(str)
    tissue = [ccle.get(c, "_UNKNOWN").split("_", 1)[1] if "_" in ccle.get(c, "_UNKNOWN")
              else "UNKNOWN" for c in cells]
    held = set(pd.read_csv(os.path.join(OUT, "primary_split_test_cell_lines.csv")).iloc[:, 0])
    d = pd.DataFrame({"depmap_id": cells, "x": xy[:, 0], "y": xy[:, 1], "tissue": tissue})
    d["cell_line"] = [str(ccle.get(c, c)).split("_")[0] for c in cells]
    d["primary_split"] = np.where(d.depmap_id.isin(held), "held out", "train or val")
    d["n_drugs"] = [int(ic.loc[c].notna().sum()) for c in cells]
    return d


def matrix_order(ic, tgt, cells):
    """A deterministic biclustering of the response matrix, shipped as source data.

    The matrix is 10.1% missing.  Column means fill the gaps for the distance
    computation only; the plotted matrix keeps its holes.
    """
    M = ic.copy()
    F = M.fillna(M.mean(axis=0))
    row_order = leaves_list(linkage(F.values, method="average", metric="correlation"))
    col_order = leaves_list(linkage(F.values.T, method="average", metric="correlation"))
    rows = [M.index[i] for i in row_order]
    cols = [M.columns[i] for i in col_order]
    t = cells.set_index("depmap_id")
    ann_r = pd.DataFrame({"depmap_id": rows, "row": range(len(rows)),
                          "tissue": [t.tissue.get(r, "UNKNOWN") for r in rows],
                          "primary_split": [t.primary_split.get(r, "") for r in rows]})
    ann_c = pd.DataFrame({"pubchem_id": cols, "col": range(len(cols)),
                          "drug_name": [tgt.drug_name.get(c, c) for c in cols],
                          "target_pathway": [tgt.TARGET_PATHWAY.get(c, "Unknown") for c in cols]})
    return M.loc[rows, cols], ann_r, ann_c


def main():
    ic, mor, tgt, cli = load()
    print(f"  response matrix {ic.shape[0]} cell lines x {ic.shape[1]} compounds, "
          f"{int(ic.notna().sum().sum()):,} observed ({ic.notna().sum().sum() / ic.size * 100:.1f}% dense)")

    d = drug_space(ic, mor, tgt)
    d.to_csv(os.path.join(SD, "Fig1_drug_chemical_space.csv"), index=False)
    print(f"  drug chemical space  {len(d)} compounds, {d.target_pathway.nunique()} target pathways")

    c = cell_space(ic, cli)
    c.to_csv(os.path.join(SD, "Fig1_cellline_space.csv"), index=False)
    print(f"  cell-line space      {len(c)} lines, {c.tissue.nunique()} tissues, "
          f"{(c.primary_split == 'held out').sum()} held out")

    M, ann_r, ann_c = matrix_order(ic, tgt, c)
    M.to_csv(os.path.join(SD, "Fig2_response_matrix_ordered.csv"))
    ann_r.to_csv(os.path.join(SD, "Fig2_response_matrix_rows.csv"), index=False)
    ann_c.to_csv(os.path.join(SD, "Fig2_response_matrix_cols.csv"), index=False)
    print(f"  ordered matrix       {M.shape[0]} x {M.shape[1]}, seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
