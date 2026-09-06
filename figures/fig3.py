#!/usr/bin/env python
"""Figure 3: the two prioritisation modes, and how close the model gets to the ceiling.

Run from the repository root, after prioritisation_modes.py,
screen_agreement_ceiling.py and stratified_performance.py.
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
FIGS = os.path.join(ROOT, "outputs", "figures")
KS = (1, 3, 5, 10)

SHORT = {"Chromatin histone acetylation": "Chromatin acetyl.",
         "Chromatin histone methylation": "Chromatin methyl.",
         "Protein stability and degradation": "Protein stability",
         "PI3K/MTOR signaling": "PI3K/MTOR", "ERK MAPK signaling": "ERK MAPK",
         "RTK signaling": "RTK", "EGFR signaling": "EGFR", "WNT signaling": "WNT",
         "Apoptosis regulation": "Apoptosis", "DNA replication": "DNA replication",
         "Genome integrity": "Genome integrity", "Other, kinases": "Other kinases"}


# Height is a consequence of the panel widths, not a page-area target: an
# ordinary panel is at most as tall as it is wide, and the leftover a
# rows_mm block would otherwise push to the bottom margin is removed here
# rather than left as white space under the figure. assert_aspect and
# assert_margins hold both.
H_FIG = 162.0


def short(name):
    import panels as _ph
    name = _ph.short_pathway(name)
    return SHORT.get(name, name)


def panel_mode1(ax, tissue, title):
    s = pd.read_csv(os.path.join(RES, "prioritisation_modes_summary.csv"))
    g = s[(s.tissue == tissue) & (s["mode"] == "rank drugs within a cell line")].sort_values("k")
    x = np.arange(len(KS))
    for col, colour, mk, ls, lab in [("bcgdrp", ns.NOMINAL[0], "o", "-", "BCGDRP"),
                                     ("drug_mean", ns.NOMINAL[1], "s", "--", "per-drug mean"),
                                     ("random", ns.NEUTRAL, "^", ":", "random")]:
        ax.plot(x, g[col].values, color=colour, marker=mk, ls=ls, ms=3.6, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontweight="bold", pad=2.4, fontsize=6.0)
    ns.legend_clear(ax, handlelength=1.8)


def panel_mode2(ax):
    d = pd.read_csv(os.path.join(RES, "prioritisation_mode2_per_drug.csv"))
    c = pd.read_csv(os.path.join(RES, "ctrpv2_drug_level_seed_aggregated.csv"))
    sets = [("Urinary tract", d[d.tissue == "URINARY_TRACT"].scc_bcgdrp, ns.NOMINAL[0], "-"),
            ("Breast", d[d.tissue == "BREAST"].scc_bcgdrp, ns.NOMINAL[1], "--"),
            ("CTRPv2 screen", c.bcgdrp, ns.NOMINAL[4], "-.")]
    for lab, v, colour, ls in sets:
        ax.hist(v, bins=26, histtype="step", color=colour, lw=1.1, ls=ls,
                label=f"{lab} (n={len(v)})")
    ax.axvline(0, color=ns.INK, lw=0.7, ls=":")
    ax.set_xlabel(r"Within-drug Spearman $\rho$")
    ax.set_ylabel("Compounds")
    ax.set_title("Within-drug ranking", fontweight="bold", pad=2.4, fontsize=6.0)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.18)
    ns.legend_clear(ax, handlelength=1.6, fontsize=5.8)


def panel_scatter(ax):
    m = pd.read_csv(os.path.join(RES, "model_vs_ceiling_per_drug.csv"))
    ax.scatter(m.ceiling_rho, m.bcgdrp, s=9, color=ns.NOMINAL[0], linewidths=0, zorder=3)
    lim = [-0.55, 0.85]
    ax.plot(lim, lim, color=ns.INK, lw=0.7, ls=":", zorder=1, label="ceiling")
    s = stats.linregress(m.ceiling_rho, m.bcgdrp)
    xs = np.linspace(m.ceiling_rho.min(), m.ceiling_rho.max(), 40)
    ax.plot(xs, s.intercept + s.slope * xs, color=ns.NOMINAL[1], lw=1.1, zorder=2)
    r = stats.spearmanr(m.ceiling_rho, m.bcgdrp)[0]
    ax.text(0.04, 0.95, f"$\\rho$ = {r:.2f}", transform=ax.transAxes, fontsize=6.4,
            va="top", color=ns.INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Screen-agreement ceiling $\rho$")
    ax.set_ylabel(r"BCGDRP $\rho$")
    ax.set_title("Model tracks ceiling", fontweight="bold", pad=2.4, fontsize=6.0)
    ns.legend_clear(ax, handlelength=1.6)


def panel_attain(ax):
    m = pd.read_csv(os.path.join(RES, "model_vs_ceiling_per_drug.csv"))
    ceil = m.ceiling_rho.mean()
    vals = [ceil, m.bcgdrp.mean(), m.bandrp_gate.mean()]
    labels = [f"Two screens\n(ceiling)", "BCGDRP", "BANDRP-gate"]
    cols = [ns.NEUTRAL, ns.NOMINAL[0], ns.NOMINAL[1]]
    ns.dotplot(ax, labels, vals, colors=cols, size=18)
    for yi, v in zip(np.arange(3)[::-1], vals):
        ax.text(v + 0.012, yi, f"{v:.3f}" + ("" if yi == 2 else f"  ({100*v/ceil:.0f}%)"),
                fontsize=5.8, va="center", color=ns.INK)
    ax.set_xlim(0, ceil * 1.55)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.set_xlabel(r"Within-drug $\rho$ against CTRPv2")
    ax.set_title("Attainment", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_pathway_ctrp(ax):
    s = pd.read_csv(os.path.join(RES, "stratified_ctrpv2_by_pathway.csv"))
    s = s.sort_values("delta")
    cols = [ns.NOMINAL[0] if (lo > 0) else (ns.NOMINAL[1] if hi < 0 else ns.NEUTRAL)
            for lo, hi in zip(s.ci_low, s.ci_high)]
    labs = [f"{short(n)} ({k})" for n, k in zip(s.stratum, s.n_drugs)]
    # the bootstrap intervals are asymmetric; draw them, not a half-width
    ns.dotplot(ax, labs, s.delta.values, colors=cols, vline=0.0,
               lo=s.ci_low.values, hi=s.ci_high.values)
    ax.set_yticklabels(labs, fontsize=5.2)
    ax.set_xlabel(r"BCGDRP $-$ BANDRP-gate  ($\rho$)")
    ax.set_title("Strata: no effect", fontweight="bold", pad=2.4, fontsize=6.0)


def panel_pathway_holdout(ax):
    h = pd.read_csv(os.path.join(RES, "stratified_holdout_by_pathway.csv"))
    piv = h.pivot_table(index="stratum", columns="tissue", values="mean_scc")
    piv = piv.dropna().sort_values("URINARY_TRACT", ascending=False).head(12)
    y = np.arange(len(piv))[::-1]
    for col, colour, mk, off in [("URINARY_TRACT", ns.NOMINAL[0], "o", 0.16),
                                 ("BREAST", ns.NOMINAL[1], "s", -0.16)]:
        ax.scatter(piv[col], y + off, s=11, color=colour, marker=mk, linewidths=0,
                   label=col.split("_")[0].title(), zorder=3)
    ax.axvline(0, color=ns.INK, lw=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([short(i) for i in piv.index], fontsize=5.2)
    ax.set_xlabel(r"Mean within-drug $\rho$")
    ax.set_title("By pathway, hold-outs", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.legend(loc="lower right", handlelength=1.0, fontsize=5.8,
              bbox_to_anchor=(1.02, -0.02))


def panel_coverage(ax):
    m = pd.read_csv(os.path.join(RES, "model_vs_ceiling_per_drug.csv"))
    ax.scatter(m.n_cells_ceiling, m.ceiling_rho, s=9, color=ns.NEUTRAL,
               linewidths=0, label="ceiling", zorder=2)
    ax.scatter(m.n_cells_ceiling, m.bcgdrp, s=9, color=ns.NOMINAL[0],
               marker="s", linewidths=0, label="BCGDRP", zorder=3)
    ax.set_xlabel("Shared cell lines per compound")
    ax.set_ylabel(r"Within-drug $\rho$")
    ax.set_title("Not a coverage effect", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.legend(loc="lower right", handlelength=1.0)


def panel_ceiling_pathway(ax):
    z = pd.read_csv(os.path.join(RES, "stratified_ceiling_by_pathway.csv")).sort_values("ceiling")
    y = np.arange(len(z))[::-1]
    ax.hlines(y, z.bcgdrp, z.ceiling, color=ns.NEUTRAL, lw=0.9, zorder=2)
    ax.scatter(z.ceiling, y, s=13, color=ns.NEUTRAL, linewidths=0, zorder=3, label="ceiling")
    ax.scatter(z.bcgdrp, y, s=13, color=ns.NOMINAL[0], marker="s", linewidths=0,
               zorder=3, label="BCGDRP")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(n)} ({k})" for n, k in zip(z.stratum, z.n_drugs)], fontsize=5.4)
    ax.set_xlabel(r"Within-drug $\rho$")
    ax.set_xlim(0, 0.6)
    ax.set_title("Ceiling by pathway", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handlelength=1.0, columnspacing=1.0, fontsize=5.8)


def panel_percell(ax):
    p = pd.read_csv(os.path.join(RES, "within_cell_scc_disease_holdouts.csv"))
    for (t, g), colour, mk in zip(p.groupby("tissue"), [ns.NOMINAL[1], ns.NOMINAL[0]], ["s", "o"]):
        ax.scatter(g.scc_drug_mean, g.scc_bcgdrp, s=11, color=colour, marker=mk,
                   linewidths=0, label=f"{t.split('_')[0].title()} (n={len(g)})", zorder=3)
    lim = [0.55, 0.95]
    ax.plot(lim, lim, color=ns.INK, lw=0.7, ls=":", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Per-drug mean, within-cell $\rho$")
    ax.set_ylabel(r"BCGDRP, within-cell $\rho$")
    ax.set_title("Within-cell ranking", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.legend(loc="upper left", handlelength=1.0, fontsize=5.8)


def hero_ceiling(fig, spec, track_mm=3.0):
    """Every compound against the agreement two independent screens achieve.

    Promoted from the supplement because it is the paper's argument in one view:
    the model tracks the ceiling, and where the screens disagree it cannot win.
    Compounds run along x so the names are legible and a target-pathway track can
    sit flush beneath them.
    """
    d = pd.read_csv(os.path.join(RES, "model_vs_ceiling_per_drug.csv")).sort_values("ceiling_rho")
    tgt = pd.read_csv(os.path.join(ROOT, "source_data",
                                   "drug_target_annotation.csv")).drop_duplicates("drug_name")
    path = tgt.set_index("drug_name").TARGET_PATHWAY
    W, H = ns.figure_mm(fig)
    b = spec.get_position(fig)
    gs = spec.subgridspec(2, 1, hspace=0.0,
                          height_ratios=[b.height * H - track_mm, track_mm])
    ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])
    x = np.arange(len(d))
    ax.vlines(x, d.bcgdrp, d.ceiling_rho, color=ns.NEUTRAL, lw=0.55, zorder=2)
    ax.scatter(x, d.ceiling_rho, s=7, color=ns.NEUTRAL, linewidths=0, zorder=3,
               label="two screens agree with each other")
    ax.scatter(x, d.bcgdrp, s=7, color=ns.NOMINAL[0], marker="s", linewidths=0, zorder=4,
               label="BCGDRP")
    ax.scatter(x, d.bandrp_gate, s=6, color=ns.NOMINAL[1], marker="^", linewidths=0,
               zorder=4, label="BANDRP-gate")
    ax.axhline(0, color=ns.INK, lw=0.5, zorder=1)
    ax.set_xlim(-1, len(d)); ax.set_xticks([])
    ax.set_ylabel(r"Within-drug Spearman $\rho$ against CTRPv2")
    series = ns.legend_clear(ax, ncol=3, fontsize=5.8, handlelength=0.9,
                             columnspacing=1.0, loc="upper right")

    top = list(path.reindex(d.drug_name).fillna("Other").value_counts().index[:6])
    cols = ns.nominal_n(6) + [ns.NEUTRAL]
    code = [top.index(v) if v in top else 6
            for v in path.reindex(d.drug_name).fillna("Other")]
    axt.imshow(np.asarray(code)[None, :], aspect="auto", interpolation="nearest",
               cmap=ListedColormap(cols),
               norm=BoundaryNorm(np.arange(-0.5, len(cols) + 0.5), len(cols)))
    axt.set_yticks([])
    axt.set_xticks(x)
    axt.set_xticklabels(d.drug_name, rotation=90, fontsize=4.4)
    axt.tick_params(length=0, pad=1.0)
    for sp in axt.spines.values():
        sp.set_visible(False)
    # the compound count and the ordering are stated in the caption; a label here
    # would sit under the rotated names and collide with the satellite titles
    # the pathway key lives inside the hero, in the empty band above the low-ceiling
    # compounds, so it needs no strip of its own and the compound names stay legible
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="s", ls="", ms=2.6, mfc=c, mec="none") for c in cols]
    lg = ax.legend(h, [panels.short_pathway(n) for n in top] + ["Remaining"],
                   title="Target pathway", loc="upper left", bbox_to_anchor=(0.005, 0.99),
                   ncol=5, frameon=False, fontsize=4.9, title_fontsize=5.2,
                   handlelength=0.7, columnspacing=0.9, handletextpad=0.3,
                   borderpad=0.0, labelspacing=0.2)
    lg.get_title().set_fontweight("bold")
    # ax.legend replaces whatever legend the axes already had, so the series key
    # has to be re-attached after the pathway key is built
    ax.add_artist(series)
    return ax, (top, cols)



def precision_heat(ax):
    """Precision at k for both hold-outs and all three rankers, values printed.

    Two line panels showed twelve numbers each and needed a legend. The same
    twenty-four numbers fit here with their identity on the axes.
    """
    d = pd.read_csv(os.path.join(RES, "prioritisation_modes_summary.csv"))
    d = d[d["mode"].str.startswith("rank drugs")]
    rows, labels = [], []
    for tis, short in (("URINARY_TRACT", "Urinary"), ("BREAST", "Breast")):
        g = d[d.tissue == tis].sort_values("k")
        for col, name in (("bcgdrp", "BCGDRP"), ("drug_mean", "per-drug mean"),
                          ("random", "random")):
            rows.append(g[col].values)
            labels.append(f"{short}, {name}")
    A = np.vstack(rows)
    cmap = LinearSegmentedColormap.from_list("m", sorted(ns.LADDER, key=ns._luminance)[::-1])
    ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(A.shape[1]))
    ax.set_xticklabels([f"k={int(v)}" for v in sorted(d.k.unique())], fontsize=5.2)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                    color="white" if A[i, j] > 0.55 else ns.INK)
    ax.set_title("Precision@k, both hold-outs", fontweight="bold", pad=2.4, fontsize=6.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def paired_difference(ax):
    """Model minus the no-model rule, with the interval and the paired test.

    This is the claim the section actually makes, and the line panels never showed
    it: the difference straddles zero at every depth in both hold-outs.
    """
    d = pd.read_csv(os.path.join(RES, "prioritisation_modes_summary.csv"))
    d = d[d["mode"].str.startswith("rank drugs")].sort_values(["tissue", "k"])
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        colour = ns.NOMINAL[0] if r.tissue == "URINARY_TRACT" else ns.NOMINAL[1]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=colour, lw=1.2,
                solid_capstyle="round", zorder=3)
        ax.scatter(r.delta, yi, s=11, color=colour, linewidths=0, zorder=4)
        ax.text(0.995, yi, f"P={r.wilcoxon_p:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=4.4, color=ns.INK)
    ax.axvline(0, color=ns.INK, lw=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'Urinary' if t == 'URINARY_TRACT' else 'Breast'}, k={int(k)}"
                        for t, k in zip(d.tissue, d.k)], fontsize=5.0)
    ax.set_xlabel("BCGDRP minus per-drug mean")
    ns.headroom(ax)          # keeps the panel letter inside, off the top interval
    ax.set_title("The model does not help here", fontweight="bold", pad=2.4, fontsize=6.0)


def main():
    fig = ns.new_figure(ns.W_2COL, H_FIG)
    # The hero drew 192 points in 17,683 mm2, which is 0.01 elements per mm2. It
    # needs enough height to separate three series and room for the compound
    # names, and nothing beyond that. The rest of the page goes to eight satellites.
    outer = ns.grid_mm(fig, 3, 1, left=24, right=3, top=4, bottom=10,
                       hgap=15.0, rows_mm=[62.0, 28.0, 28.0])
    ax = {}
    ax["a"], _ = hero_ceiling(fig, outer[0], track_mm=2.6)

    W, _ = ns.figure_mm(fig)
    span = W - 24 - 3
    jobs = [
        ("b", precision_heat),
        ("c", paired_difference),
        ("d", panel_percell), ("e", panel_mode2),
        ("f", panel_attain), ("g", panel_scatter),
        ("h", panel_pathway_ctrp), ("i", panel_pathway_holdout),
    ]
    for row in (0, 1):
        gs = outer[1 + row].subgridspec(1, 4, wspace=13.0 / ((span - 39.0) / 4))
        for c in range(4):
            L, fn = jobs[row * 4 + c]
            ax[L] = fig.add_subplot(gs[0, c]); fn(ax[L])

    for L, a in ax.items():
        ns.panel_label(fig, a, L)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(FIGS, "fig3.pdf"), ns.W_2COL, H_FIG)
    plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
