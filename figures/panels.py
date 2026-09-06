#!/usr/bin/env python
"""The response-matrix hero panel, with annotation tracks flush against it.

The paper's central quantity is that compound identity carries 67.4% of the
variance in the response matrix. That is a number a reader has to take on trust
unless the matrix is shown, at which point the banding by compound is visible and
the per-drug-mean baseline stops looking like a strawman.

Drugs are rows and cell lines are columns, which is the convention in this
literature, so the target-pathway track runs down the left and the tissue track
runs along the top. Tracks are flush: no gap, because a gap would read as a
separate panel rather than as an annotation of this one.
"""
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

import style as ns

# GDSC and CCLE spell these out; a key strip needs them short and still recognisable.
TISSUE_SHORT = {
    "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "Haematopoietic",
    "CENTRAL_NERVOUS_SYSTEM": "CNS", "LARGE_INTESTINE": "Large intestine",
    "UPPER_AERODIGESTIVE_TRACT": "Aerodigestive", "AUTONOMIC_GANGLIA": "Autonomic ganglia",
    "URINARY_TRACT": "Urinary tract", "SOFT_TISSUE": "Soft tissue",
}
PATHWAY_SHORT = {
    "PI3K/MTOR signaling": "PI3K/MTOR", "RTK signaling": "RTK", "ERK MAPK signaling": "ERK MAPK",
    "Other, kinases": "Other kinases", "Apoptosis regulation": "Apoptosis",
    "Chromatin histone acetylation": "Chromatin acetyl.",
    "Chromatin histone methylation": "Chromatin methyl.", "Chromatin other": "Chromatin other",
}


def short_tissue(t):
    return TISSUE_SHORT.get(t, t.replace("_", " ").title())


def short_pathway(p):
    return PATHWAY_SHORT.get(p, p)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "source_data")

CMAP = LinearSegmentedColormap.from_list("resp", sorted(ns.LADDER, key=ns._luminance))
CMAP.set_bad("#FFFFFF")


def load_matrix():
    M = pd.read_csv(os.path.join(SD, "Fig2_response_matrix_ordered.csv"), index_col=0)
    rows = pd.read_csv(os.path.join(SD, "Fig2_response_matrix_rows.csv"))
    cols = pd.read_csv(os.path.join(SD, "Fig2_response_matrix_cols.csv"))
    return M, rows, cols


def _track(ax, codes, colours, horizontal):
    arr = np.asarray(codes)[None, :] if horizontal else np.asarray(codes)[:, None]
    ax.imshow(arr, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(colours), norm=BoundaryNorm(
                  np.arange(-0.5, len(colours) + 0.5), len(colours)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw(fig, spec, top_mm=3.4, left_mm=3.4, n_pathways=6, n_tissues=6):
    """Draw the hero into `spec`. Returns (matrix axes, legend handles)."""
    M, rows, cols = load_matrix()
    A = M.values.T                                   # drugs x cell lines
    W, H = ns.figure_mm(fig)
    b = spec.get_position(fig)
    cell_w, cell_h = b.width * W, b.height * H

    gs = spec.subgridspec(2, 2, wspace=0.0, hspace=0.0,
                          height_ratios=[top_mm, cell_h - top_mm],
                          width_ratios=[left_mm, cell_w - left_mm])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 0]).axis("off")

    # A 2-98 clip leaves the matrix washed out; 8-92 keeps the compound banding
    # legible, which is the one thing this panel exists to show.
    lo, hi = np.nanpercentile(A, [8, 92])
    im = ax.imshow(np.ma.masked_invalid(A), aspect="auto", interpolation="nearest",
                   cmap=CMAP, vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{A.shape[1]} cell lines, biclustered", labelpad=1.5)
    # the y label belongs on the track axes, otherwise it is drawn over the track
    ax_l.set_ylabel(f"{A.shape[0]} compounds", labelpad=2.0)

    def codes(series, k):
        top = list(series.value_counts().index[:k])
        cmap = dict(zip(top, ns.nominal_n(k)))
        c = [top.index(v) if v in top else k for v in series]
        return c, list(cmap.values()) + [ns.NEUTRAL], top

    pc, pcol, ptop = codes(cols.target_pathway, n_pathways)
    tc, tcol, ttop = codes(rows.tissue, n_tissues)
    _track(ax_l, pc, pcol, horizontal=False)
    _track(ax_t, tc, tcol, horizontal=True)
    return ax, (ptop, pcol, ttop, tcol), im
