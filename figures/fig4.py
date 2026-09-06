#!/usr/bin/env python
"""Figure 4: what each molecular modality contributes, and where the held-out cells sit.

The modality ablations elsewhere in the paper remove an input from BCGDRP, so they
measure the modality and the encoder together.  These panels measure the modalities
against one another on a common task and a common split, using the same simple
estimator for each, so the comparison between molecular layers does not depend on
any particular architecture.

Run from the repository root, after modality_information.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import (BoundaryNorm, LinearSegmentedColormap,
                               ListedColormap)

import style as ns
import panels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "analysis")
SD = os.path.join(ROOT, "source_data")
FIGS = os.path.join(ROOT, "outputs", "figures")

ORDER = ["expression", "pathway", "methylation", "mutation"]
COL = {"expression": ns.NOMINAL[0], "methylation": ns.NOMINAL[1],
       "mutation": ns.NOMINAL[2], "pathway": ns.NOMINAL[4],
       "expression+methylation": ns.NOMINAL[3], "all four": ns.NEUTRAL}
SHORT = {"Chromatin histone acetylation": "Chromatin acetyl.",
         "Chromatin histone methylation": "Chromatin methyl.",
         "Protein stability and degradation": "Protein stability",
         "PI3K/MTOR signaling": "PI3K/MTOR", "ERK MAPK signaling": "ERK MAPK",
         "RTK signaling": "RTK", "EGFR signaling": "EGFR", "WNT signaling": "WNT",
         "Apoptosis regulation": "Apoptosis", "Other, kinases": "Other kinases"}


# Height is a consequence of the panel widths, not a page-area target: an
# ordinary panel is at most as tall as it is wide, and the leftover a
# rows_mm block would otherwise push to the bottom margin is removed here
# rather than left as white space under the figure. assert_aspect and
# assert_margins hold both.
H_FIG = 186.0


def _ridge():
    return pd.read_csv(os.path.join(RES, "modality_within_drug_ridge.csv"))


def panel_a(ax):
    """Distribution over compounds of the information each modality carries."""
    d = _ridge()
    groups = ORDER + ["expression+methylation", "all four"]
    for i, mod in enumerate(groups):
        v = d[d.modality == mod].spearman.values
        if not len(v):
            continue
        parts = ax.violinplot([v], positions=[i], widths=0.72, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COL.get(mod, ns.NEUTRAL)); b.set_alpha(0.55); b.set_linewidth(0)
        ax.scatter([i], [v.mean()], s=14, color=ns.INK, zorder=4, marker="_", linewidths=1.2)
        ax.text(i, 1.02, f"{v.mean():.2f}", ha="center", fontsize=5.2, color=ns.INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([{"expression": "exp", "pathway": "path",
                         "methylation": "meth", "mutation": "mut",
                         "expression+methylation": "e+m",
                         "all four": "all 4"}.get(m, m[:5])
                        for m in groups], rotation=0, fontsize=5.0)
    ax.axhline(0, color=ns.INK, lw=0.6, ls=":")
    ax.set_ylabel(r"Within-drug Spearman $\rho$")
    ax.set_ylim(-0.15, 1.12)
    ax.set_title("Information per modality", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_b(ax):
    """Expression against methylation, paired within compound."""
    w = _ridge().pivot_table(index="drug_name", columns="modality", values="spearman")
    b = w[["expression", "methylation"]].dropna()
    ax.scatter(b.methylation, b.expression, s=7, color=ns.NOMINAL[0], linewidths=0, zorder=3)
    lim = [0.0, 0.95]
    ax.plot(lim, lim, color=ns.INK, lw=0.7, ls=":", zorder=1)
    W, p = stats.wilcoxon(b.expression, b.methylation)
    ax.text(0.04, 0.96, f"expression higher for\n{int((b.expression>b.methylation).sum())}/{len(b)} compounds\n"
                        f"$P$ = {p:.0e}",
            transform=ax.transAxes, fontsize=5.4, va="top", color=ns.INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Methylation $\rho$")
    ax.set_ylabel(r"Expression $\rho$")
    ax.set_title("Expression carries more", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_c(ax):
    """Does adding methylation to expression help?"""
    w = _ridge().pivot_table(index="drug_name", columns="modality", values="spearman")
    t = w[["expression", "methylation", "expression+methylation"]].dropna()
    gain = t["expression+methylation"] - t[["expression", "methylation"]].max(axis=1)
    ax.hist(gain, bins=30, color=ns.NOMINAL[4], alpha=0.75, linewidth=0)
    ax.axvline(0, color=ns.INK, lw=0.8, ls=":")
    ax.text(0.04, 0.96, f"mean {gain.mean():+.3f}\nimproves {int((gain>0).sum())}/{len(t)}",
            transform=ax.transAxes, fontsize=5.4, va="top", color=ns.INK)
    ax.set_xlabel("Change from combining the two")
    ax.set_ylabel("Compounds")
    ax.set_title("The two layers are redundant", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_pca(ax, modality, label):
    a = pd.read_csv(os.path.join(RES, f"modality_pca_{modality}_train.csv"), index_col=0)
    b = pd.read_csv(os.path.join(RES, f"modality_pca_{modality}_test.csv"), index_col=0)
    ax.scatter(a.pc1, a.pc2, s=3.2, color=ns.NEUTRAL, alpha=0.45, linewidths=0,
               label=f"training ({len(a)})", zorder=2, rasterized=True)
    ax.scatter(b.pc1, b.pc2, s=9, color=COL[modality], marker="s", linewidths=0,
               label=f"held out ({len(b)})", zorder=3)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{label} feature space", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.legend(loc="upper right", handlelength=0.9, fontsize=5.4, borderpad=0.1)


def panel_distance(ax):
    d = pd.read_csv(os.path.join(RES, "modality_feature_distance.csv"))
    for i, mod in enumerate(ORDER):
        v = d[d.modality == mod].nn_distance.values
        if not len(v):
            continue
        v = v / np.median(v)
        parts = ax.violinplot([v], positions=[i], widths=0.72, showextrema=False)
        for bb in parts["bodies"]:
            bb.set_facecolor(COL[mod]); bb.set_alpha(0.55); bb.set_linewidth(0)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([m[:4] for m in ORDER], rotation=0, fontsize=5.4)
    ax.set_ylabel("Distance to nearest\ntraining cell (median = 1)")
    ax.set_title("Held-out cells are in distribution", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_dist_error(ax):
    """Distance is computed within each hold-out against that hold-out's own training set."""
    d = pd.read_csv(os.path.join(RES, "modality_holdout_distance_vs_error.csv"))
    d = d[d.modality == "expression"]
    su = pd.read_csv(os.path.join(RES, "modality_holdout_distance_summary.csv"))
    su = su[su.modality == "expression"].set_index("tissue")
    for (tis, g), colour, mk in zip(d.groupby("tissue"),
                                    [ns.NOMINAL[1], ns.NOMINAL[0]], ["s", "o"]):
        x = g.nn_distance / g.nn_distance.median()
        ax.scatter(x, g.rmse, s=11, color=colour, marker=mk, linewidths=0, zorder=3,
                   label=f"{tis.split('_')[0].title()} "
                         f"($\\rho$={su.loc[tis, 'spearman']:+.2f}, "
                         f"$P$={su.loc[tis, 'p_value']:.3f})")
        if len(g) > 8:
            sl = stats.linregress(x, g.rmse)
            xs = np.linspace(x.min(), x.max(), 30)
            ax.plot(xs, sl.intercept + sl.slope * xs, color=colour, lw=1.0,
                    ls=["-", "--"][mk == "s"], zorder=2)
    ax.set_xlabel("Distance to nearest training cell\n(expression, median = 1)")
    ax.set_ylabel("Per-cell RMSE")
    ax.set_title("Error grows with molecular distance", fontweight="bold", pad=2.4, fontsize=6.0)
    # The worst-predicted cell line sits at RMSE 3.41, and it is the point this panel
    # most needs to show. Open headroom above the data so the legend cannot cover it.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, d.rmse.max() * 1.30)
    ax.legend(loc="upper left", handlelength=1.0, fontsize=5.0, borderpad=0.1,
              labelspacing=0.2, framealpha=0.0)


def panel_pathway(ax):
    g = pd.read_csv(os.path.join(RES, "modality_by_pathway.csv")).sort_values("expression")
    y = np.arange(len(g))[::-1]
    ax.hlines(y, g.methylation, g.expression, color=ns.NEUTRAL, lw=0.8, zorder=2)
    ax.scatter(g.expression, y, s=11, color=ns.NOMINAL[0], marker="o", linewidths=0,
               zorder=3, label="expression")
    ax.scatter(g.methylation, y, s=11, color=ns.NOMINAL[1], marker="s", linewidths=0,
               zorder=3, label="methylation")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SHORT.get(n, n)} ({k})" for n, k in zip(g.stratum, g.n)], fontsize=4.8)
    ax.set_xlabel(r"Mean within-drug $\rho$")
    ax.set_title("By compound target pathway", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_xlim(0.36, 0.80)
    ns.legend_clear(ax, handlelength=1.0, fontsize=5.2, borderpad=0.1,
                    labelspacing=0.2)


def panel_features(ax):
    n_feat = {"expression": 714, "methylation": 603, "mutation": 715, "pathway": 1283}
    d = _ridge()
    xs = [n_feat[m] for m in ORDER]
    ys = [d[d.modality == m].spearman.mean() for m in ORDER]
    for x, y, m in zip(xs, ys, ORDER):
        ax.scatter(x, y, s=26, color=COL[m], linewidths=0, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=5.2, color=ns.INK)
    ax.set_xlabel("Features in the modality")
    ax.set_ylabel(r"Mean within-drug $\rho$")
    ax.set_ylim(0, 0.78)
    ax.set_xlim(400, 1500)
    ax.set_title("Not a feature-count effect", fontweight="bold", pad=2.4, fontsize=6.0)


def hero_modality_matrix(fig, spec, track_mm=3.0):
    """Every compound, every molecular layer, one cell each.

    The claim this figure makes is that gene expression carries most of the
    recoverable response information. Aggregates can hide a minority of compounds
    where that fails, so the per-compound value is shown for all 169 rather than
    summarised.
    """
    d = pd.read_csv(os.path.join(RES, "modality_within_drug_ridge.csv"))
    piv = d.pivot_table(index="modality", columns="drug_name", values="spearman")
    order = [m for m in ORDER if m in piv.index] + \
            [m for m in piv.index if m not in ORDER]
    piv = piv.loc[order]
    piv = piv[piv.loc[ORDER[0]].sort_values(ascending=False).index]

    tgt = pd.read_csv(os.path.join(ROOT, "source_data",
                                   "drug_target_annotation.csv")).drop_duplicates("drug_name")
    path = tgt.set_index("drug_name").TARGET_PATHWAY.reindex(piv.columns).fillna("Other")

    W, H = ns.figure_mm(fig)
    b = spec.get_position(fig)
    gs = spec.subgridspec(2, 1, hspace=0.0,
                          height_ratios=[b.height * H - track_mm, track_mm])
    ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])
    cmap = LinearSegmentedColormap.from_list("m", sorted(ns.LADDER, key=ns._luminance)[::-1])
    im = ax.imshow(piv.values, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0.0, vmax=0.85)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([m.replace("expression", "exp").replace("methylation", "meth")
                    .replace("+", " + ") for m in piv.index], fontsize=6.0)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)

    top = list(path.value_counts().index[:6])
    cols = ns.nominal_n(6) + [ns.NEUTRAL]
    code = [top.index(v) if v in top else 6 for v in path]
    axt.imshow(np.asarray(code)[None, :], aspect="auto", interpolation="nearest",
               cmap=ListedColormap(cols),
               norm=BoundaryNorm(np.arange(-0.5, len(cols) + 0.5), len(cols)))
    axt.set_yticks([]); axt.set_xticks([])
    for sp in axt.spines.values():
        sp.set_visible(False)
    axt.set_xlabel(f"{piv.shape[1]} compounds, ordered by the expression value", labelpad=1.5)
    return ax, im, (top, cols)


def main():
    fig = ns.new_figure(ns.W_2COL, H_FIG)
    # The hero drew 1,014 values in 16,964 mm2, which is 0.06 elements per mm2,
    # ninety times thinner than the response matrix in Figure 2. Six rows need
    # about 3 mm each, so 20 mm, and the 88 mm that frees goes to nine satellites.
    outer = ns.grid_mm(fig, 3, 1, left=19, right=3, top=4, bottom=13,
                       hgap=6.0, rows_mm=[20.0, 5.0, 132.0])
    ax = {}
    ax["a"], im, (ptop, pcol) = hero_modality_matrix(fig, outer[0], track_mm=2.2)

    keys = fig.add_subplot(outer[1]); keys.axis("off")
    kp = keys.get_position()
    cax = fig.add_axes([kp.x0, kp.y0 + kp.height * 0.30, 0.070, kp.height * 0.42])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"Within-drug $\rho$", fontsize=4.8, labelpad=0.8)
    cb.ax.tick_params(labelsize=4.4, length=1.2, pad=0.8)
    cb.outline.set_linewidth(0.4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="s", ls="", ms=2.4, mfc=c, mec="none") for c in pcol]
    lg = keys.legend(h, [panels.short_pathway(n) for n in ptop] + ["Remaining"],
                     title="Target pathway", loc="upper left", bbox_to_anchor=(0.135, 1.55),
                     ncol=9, frameon=False, fontsize=4.5, title_fontsize=4.9,
                     handlelength=0.6, columnspacing=0.7, handletextpad=0.25, borderpad=0.0)
    lg.get_title().set_fontweight("bold")

    W, _ = ns.figure_mm(fig)
    span = W - 19 - 3
    jobs = [("b", panel_a), ("c", panel_b), ("d", panel_c),
            ("e", lambda a: panel_pca(a, "expression", "Expression")),
            ("f", lambda a: panel_pca(a, "methylation", "Methylation")),
            ("g", panel_distance), ("h", panel_dist_error),
            ("i", panel_pathway), ("j", panel_features)]
    # the satellites need their own row spacing: rotated tick labels plus the next
    # row's title do not fit in the 6 mm that separates the hero from its key strip
    sat = outer[2].subgridspec(3, 3, wspace=14.0 / ((span - 28.0) / 3),
                               hspace=16.0 / ((132.0 - 32.0) / 3))
    for i, (L, fn) in enumerate(jobs):
        ax[L] = fig.add_subplot(sat[i // 3, i % 3]); fn(ax[L])

    for L, a in ax.items():
        ns.panel_label(fig, a, L)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(FIGS, "fig4.pdf"), ns.W_2COL, H_FIG)
    plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
