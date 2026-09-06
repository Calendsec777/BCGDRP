#!/usr/bin/env python
"""Figure 6: the two disease-focused hold-outs.

Every cell line carrying the tissue annotation is withheld, so the panels show
transfer to a tissue group whose drug-response profiles were entirely absent from
model development.  The per-drug-mean predictor is drawn alongside wherever it is
defined, because the two hold-outs disagree about whether the model beats it.

Run from the repository root.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as ns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "source_data")
RES = os.path.join(ROOT, "results", "analysis")
FIGS = os.path.join(ROOT, "outputs", "figures")
TIS = [("URINARY_TRACT", "Urinary tract", ns.NOMINAL[0], "o"),
       ("BREAST", "Breast", ns.NOMINAL[1], "s")]
# a sequential map must be monotonic in luminance, so order the ladder by it first;
# dark is sensitive, pale is resistant
_ORDERED = sorted(ns.LADDER, key=ns._luminance)
CMAP = LinearSegmentedColormap.from_list("resp", _ORDERED)


# Height is a consequence of the panel widths, not a page-area target: an
# ordinary panel is at most as tall as it is wide, and the leftover a
# rows_mm block would otherwise push to the bottom margin is removed here
# rather than left as white space under the figure. assert_aspect and
# assert_margins hold both.
H_FIG = 152.0


def _diag():
    return pd.read_csv(os.path.join(SD, "Fig2_case_prediction_diagnostics.csv"))


def scatter(ax, case, label, colour):
    d = _diag()
    d = d[d.case == case]
    ax.scatter(d.pred_logIC50, d.true_logIC50, s=0.8, alpha=0.12, color=colour,
               linewidths=0, rasterized=True)
    lim = [-8, 10]
    ax.plot(lim, lim, color=ns.INK, lw=0.6, ls=":", zorder=2)
    m = pd.read_csv(os.path.join(RES, "naive_vs_bcgdrp_disease_holdouts.csv"))
    r = m[(m.tissue == case) & (m.predictor == "BCGDRP")].iloc[0]
    b = m[(m.tissue == case) & (m.predictor == "NaiveDrugMeanPredictor")].iloc[0]
    ax.text(0.04, 0.96, f"BCGDRP  RMSE {r.RMSE:.3f}\nper-drug mean  {b.RMSE:.3f}\n$n$ = {int(r.n):,} pairs",
            transform=ax.transAxes, fontsize=5.6, va="top", color=ns.INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Predicted ln($IC_{50}$)")
    ax.set_ylabel(r"Observed ln($IC_{50}$)")
    ax.set_title(f"{label} hold-out", fontweight="bold", pad=2.4, fontsize=6.0)


def residuals(ax):
    d = _diag()
    for case, label, colour, _ in TIS:
        v = d[d.case == case].residual
        ax.hist(v, bins=44, histtype="step", color=colour, lw=1.1,
                ls=["-", "--"][case == "BREAST"], density=True,
                label=f"{label} (s.d. {v.std():.2f})")
    ax.axvline(0, color=ns.INK, lw=0.7, ls=":")
    ax.set_xlabel("Residual, predicted $-$ observed")
    ax.set_ylabel("Density")
    ax.set_xlim(-8, 8)
    ax.set_title("Residual distribution", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    ax.legend(loc="upper left", handlelength=1.6, fontsize=5.6, borderpad=0.1)


def ranked(ax, csv, label, colour):
    d = pd.read_csv(os.path.join(SD, csv)).sort_values("mean_pred_logIC50")
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d.mean_pred_logIC50, d.mean_true_logIC50, color=ns.NEUTRAL, lw=0.7, zorder=2)
    ax.scatter(d.mean_pred_logIC50, y, s=13, color=colour, marker="o",
               linewidths=0, zorder=3, label="predicted")
    ax.scatter(d.mean_true_logIC50, y, s=13, color=ns.INK, marker="x",
               linewidths=0.9, zorder=3, label="observed")
    ax.set_yticks(y)
    ax.set_yticklabels(d.drug_name, fontsize=5.2)
    ax.set_xlabel(r"Mean ln($IC_{50}$)")
    ax.set_title(f"{label}, ten most sensitive", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_ylim(-1.35, len(d) - 0.4)
    ax.legend(loc="lower right", handlelength=1.0, fontsize=5.4,
              ncol=2, columnspacing=0.8, borderpad=0.1)



def shared_response_limits(csvs, n_drugs=14):
    """One colour scale across every landscape panel, over exactly the cells each will draw."""
    lo, hi = [], []
    for c in csvs:
        d = pd.read_csv(os.path.join(SD, c))
        q = d.pivot_table(index="drug_name", columns="cell_line", values="pred_logIC50")
        q = q.loc[q.mean(axis=1).sort_values().index[:n_drugs]]
        lo.append(float(q.values.min())); hi.append(float(q.values.max()))
    return min(lo), max(hi)


def landscape(ax, csv, label, n_drugs=14, vlim=None, cbar=True):
    d = pd.read_csv(os.path.join(SD, csv))
    p = d.pivot_table(index="drug_name", columns="cell_line", values="pred_logIC50")
    order = p.mean(axis=1).sort_values().index[:n_drugs]
    p = p.loc[order]
    p = p[p.mean(axis=0).sort_values().index]
    # Both hold-out landscapes must share one colour scale, or the same colour means a
    # different response in each panel and a reader comparing them is misled.
    vmin, vmax = vlim if vlim else (None, None)
    im = ax.imshow(p.values, aspect="auto", cmap=CMAP, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(p.index, fontsize=4.4)
    if len(p.columns) <= 16:
        ax.set_xticks(range(len(p.columns)))
        ax.set_xticklabels(p.columns, fontsize=5.0, rotation=90)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"{len(p.columns)} held-out cell lines, ordered by mean predicted response",
                      fontsize=5.2)
    ax.set_title(f"{label} response landscape", fontweight="bold", pad=2.4, fontsize=6.0)
    if cbar:
        # both landscapes share one scale, so one key serves them both
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cb.ax.tick_params(labelsize=5.0)
        cb.set_label(r"Pred. ln($IC_{50}$)", fontsize=5.0, labelpad=1.0)
    for s in ax.spines.values():
        s.set_visible(True)


def percell(ax):
    p = pd.read_csv(os.path.join(RES, "within_cell_scc_disease_holdouts.csv"))
    for case, label, colour, mk in TIS:
        g = p[p.tissue == case].sort_values("scc_bcgdrp")
        ax.scatter(g.scc_bcgdrp, np.arange(len(g)) / max(len(g) - 1, 1),
                   s=10, color=colour, marker=mk, linewidths=0, label=f"{label} (n={len(g)})")
    ax.set_xlabel(r"Within-cell-line Spearman $\rho$")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Every held-out cell line", fontweight="bold", pad=2.4, fontsize=6.0)
    ns.legend_clear(ax, handlelength=1.0, fontsize=5.6)


def coverage(ax):
    s = pd.read_csv(os.path.join(SD, "Fig1_split_counts.csv"))
    c = pd.read_csv(os.path.join(SD, "Fig1_pair_coverage.csv"))
    num = s.select_dtypes("number")
    labels = [str(v) for v in s.iloc[:, 0]]
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    for col, colour in zip(num.columns, [ns.NOMINAL[0], ns.NOMINAL[1], ns.NOMINAL[2]]):
        ax.barh(y, num[col].values, left=left, height=0.55, color=colour,
                label=col, linewidth=0)
        left += num[col].values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("Cell lines")
    ax.set_title("Hold-out composition", fontweight="bold", pad=2.4, fontsize=6.0)
    ns.legend_clear(ax, handlelength=0.9, fontsize=5.4, labelspacing=0.2,
                    framealpha=0.9, facecolor="white", edgecolor="none")


def main():
    fig = ns.new_figure(ns.W_2COL, H_FIG)
    # The landscapes resolve fourteen compounds, so about 3 mm per row is enough.
    # They were given 6,300 mm2 each at 0.03 elements per mm2; now they take the
    # top band and eight satellites fill the rest.
    outer = ns.grid_mm(fig, 3, 1, left=24, right=12, top=4, bottom=11,
                       hgap=15.0, rows_mm=[48.0, 25.0, 34.0])
    ax = {}
    W, _ = ns.figure_mm(fig)
    span = W - 24 - 12
    vlim = shared_response_limits(["Fig4_urinary_tract_heatmap_values.csv",
                                   "Fig4_breast_heatmap_values.csv"])
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.5],
                               wspace=20.0 / ((span - 20.0) / 2))
    ax["a"] = fig.add_subplot(top[0, 0])
    landscape(ax["a"], "Fig4_urinary_tract_heatmap_values.csv", "Urinary tract",
              vlim=vlim, cbar=False)
    ax["b"] = fig.add_subplot(top[0, 1])
    landscape(ax["b"], "Fig4_breast_heatmap_values.csv", "Breast", vlim=vlim)

    jobs = [("c", lambda a: scatter(a, "URINARY_TRACT", "Urinary tract", ns.NOMINAL[0])),
            ("d", lambda a: scatter(a, "BREAST", "Breast", ns.NOMINAL[1])),
            ("e", residuals), ("f", percell),
            ("g", lambda a: ranked(a, "Fig4_urinary_top_ranked_drugs.csv",
                                   "Urinary tract", ns.NOMINAL[0])),
            ("h", lambda a: ranked(a, "Fig4_breast_top_ranked_drugs.csv",
                                   "Breast", ns.NOMINAL[1])),
            ("i", coverage)]
    for row in (0, 1):
        n = 4 if row == 0 else 3
        gs = outer[1 + row].subgridspec(1, n, wspace=14.0 / ((span - 14.0 * (n - 1)) / n))
        for c in range(n):
            k = row * 4 + c
            if k >= len(jobs):
                break
            L, fn = jobs[k]
            ax[L] = fig.add_subplot(gs[0, c]); fn(ax[L])

    for L, a in ax.items():
        ns.panel_label(fig, a, L)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(FIGS, "fig6.pdf"), ns.W_2COL, H_FIG)
    plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
