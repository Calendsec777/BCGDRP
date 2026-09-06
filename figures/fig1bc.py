#!/usr/bin/env python
"""Regenerate the two data-driven panels of Figure 1.

Panels a, d, e and f are author-drawn schematics. Panels b and c are regenerated
here as vector PDFs authored at their final size.

They show the two input modalities as themselves. A paper that predicts response
from compound structure and cellular state should show the reader what those
spaces look like and where the held-out cells sit in one of them, rather than
only scoring them. Split counts moved to the supplement to make room.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as ns
import panels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "source_data")
FIGS = os.path.join(ROOT, "outputs", "figures")

W_MM, H_MM = 57.0, 42.0


def panel_b():
    """169 compounds in Morgan fingerprint space, laid out by Tanimoto distance."""
    fig = ns.new_figure(W_MM, H_MM)
    ax = fig.add_axes([0.045, 0.055, 0.945, 0.900])
    d = pd.read_csv(os.path.join(SD, "Fig1_drug_chemical_space.csv"))
    # only three fit as inline labels at this size, so only three are named and
    # everything else is neutral. Five colours with three labels would leave two
    # categories that a reader cannot identify at all.
    top = list(d.target_pathway.value_counts().index[:3])
    cols = ns.nominal_n(3) + [ns.NEUTRAL]
    for i, p in enumerate(top + ["other"]):
        g = d[d.target_pathway == p] if p != "other" else d[~d.target_pathway.isin(top)]
        ax.scatter(g.x, g.y, s=5.5, color=cols[i], marker=ns.MARKERS[i],
                   linewidths=0, zorder=3 if p != "other" else 2)
    for i, p in enumerate(top):
        g = d[d.target_pathway == p]
        ax.text(g.x.median(), g.y.median(), panels.short_pathway(p), fontsize=5.0,
                color=cols[i], ha="center", va="center", fontweight="bold", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(d)} compounds, Morgan space", fontsize=5.4, labelpad=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ns.save(fig, os.path.join(FIGS, "fig1b.pdf"), W_MM, H_MM, png=False)
    plt.close(fig)


def panel_c():
    """536 cell lines in expression space, with the held-out groups marked."""
    fig = ns.new_figure(W_MM, H_MM)
    ax = fig.add_axes([0.045, 0.055, 0.945, 0.900])
    c = pd.read_csv(os.path.join(SD, "Fig1_cellline_space.csv"))
    base = c[c.primary_split != "held out"]
    ax.scatter(base.x, base.y, s=3.6, color="#D8DCE3", marker=".",
               linewidths=0, zorder=1,
               label="training and validation")
    for tis, colour, marker, lab in (
        ("URINARY_TRACT", ns.NOMINAL[3], "o", "urinary tract"),
        ("BREAST", ns.NOMINAL[0], "s", "breast"),
    ):
        g = c[c.tissue == tis]
        ax.scatter(g.x, g.y, s=7.0, color=colour, marker=marker,
                   linewidths=0, zorder=3, label=lab)
    held = c[(c.primary_split == "held out") & (~c.tissue.isin(["URINARY_TRACT", "BREAST"]))]
    ax.scatter(held.x, held.y, s=6.0, color=ns.INK, marker="x", linewidths=0.45,
               zorder=4, label="held out, primary split")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{len(c)} cell lines, expression space", fontsize=5.4, labelpad=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ns.legend_clear(ax, fontsize=5.0, handlelength=0.7, handletextpad=0.25,
                    labelspacing=0.18, borderpad=0.0, markerscale=1.1)
    ns.save(fig, os.path.join(FIGS, "fig1c.pdf"), W_MM, H_MM, png=False)
    plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    panel_b()
    panel_c()
