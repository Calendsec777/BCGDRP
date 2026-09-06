#!/usr/bin/env python
"""Figure 2: what a cell-line-disjoint model adds over response statistics.

Run from the repository root, after the scripts in `experiments/analysis/`.
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
import style as ns
import panels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "analysis")
FIGS = os.path.join(ROOT, "outputs", "figures")
CASES = {
    "Urinary tract": os.path.join(ROOT, "results", "cases", "urinary_predictions.csv"),
    "Breast": os.path.join(ROOT, "results", "cases", "breast_predictions.csv"),
}
BLIND = [("tCNNs", 1.7133, 0.7875), ("DeepCDR", 1.4729, 0.8521), ("DeepTTA", 1.4218, 0.8580),
         ("GADRP", 1.4239, 0.8586), ("BANDRP", 1.3771, 0.8676), ("BCGDRP", 1.3644, 0.8706)]
RANDOM = [("tCNNs", 1.1667, 0.9029), ("DeepCDR", 1.0296, 0.9252), ("DeepTTA", 0.9752, 0.9081),
          ("GADRP", 1.0070, 0.9283), ("BANDRP", 0.9439, 0.9366), ("BCGDRP", 0.9755, 0.9311)]


# Height is a consequence of the panel widths, not a page-area target: an
# ordinary panel is at most as tall as it is wide, and the leftover a
# rows_mm block would otherwise push to the bottom margin is removed here
# rather than left as white space under the figure. assert_aspect and
# assert_margins hold both.
H_FIG = 173.0


def panel_a(ax):
    """Variance decomposition, observed against what the model reproduces."""
    v = pd.read_csv(os.path.join(RES, "variance_decomposition.csv"))
    rows = [("GDSC2 matrix", v.iloc[0]),
            ("Urinary, obs.", v[(v.setting.str.startswith("URINARY")) &
                                            (v.quantity == "observed response")].iloc[0]),
            ("Urinary, model", v[(v.setting.str.startswith("URINARY")) &
                                          (v.quantity == "BCGDRP prediction")].iloc[0]),
            ("Breast, obs.", v[(v.setting.str.startswith("BREAST")) &
                                     (v.quantity == "observed response")].iloc[0]),
            ("Breast, model", v[(v.setting.str.startswith("BREAST")) &
                                   (v.quantity == "BCGDRP prediction")].iloc[0])]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for key, colour, lab in [("drug_pct", ns.NOMINAL[0], "compound"),
                             ("cell_pct", ns.NOMINAL[1], "cell line"),
                             ("interaction_pct", ns.NOMINAL[3], "interaction")]:
        w = np.array([r[key] for _, r in rows], float)
        ax.barh(y, w, left=left, height=0.62, color=colour, label=lab, linewidth=0)
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlabel("Share of variance (%)")
    ax.set_xlim(0, 100)
    # The bars fill the axes from 0 to 100, so no interior corner is ever clear and
    # legend_clear had nowhere to go but the upper right, where it overhung into the
    # next panel's tick labels. Open a band above the bars and put the key in it.
    ax.set_ylim(-0.7, len(rows) - 1 + 1.35)
    ax.set_title("Variance decomposition", fontweight="bold", pad=3)
    # Two columns, measured at 21.9 mm against the panel's 31 mm. Three columns
    # measure 30.9 and therefore end flush with the right edge, where the next
    # panel's tick labels begin, and the two run together.
    ax.legend(loc="upper left", ncol=2, fontsize=5.0, handlelength=0.8,
              columnspacing=0.7, handletextpad=0.3, borderpad=0.1,
              labelspacing=0.25, frameon=False, bbox_to_anchor=(0.0, 1.03))


def panel_b(ax):
    """Every model against the two response-statistics predictors, primary split."""
    p = pd.read_csv(os.path.join(RES, "naive_baselines_primary_split.csv")).set_index("predictor")
    dm = p.loc["NaiveDrugMeanPredictor", "RMSE"]
    labels = ["Global mean", "Per-drug mean"] + [b[0] for b in BLIND]
    vals = [p.loc["NaivePredictor", "RMSE"], dm] + [b[1] for b in BLIND]
    cols = [ns.NEUTRAL, ns.NEUTRAL] + [ns.NOMINAL[1] if v > dm else ns.NOMINAL[0] for v in vals[2:]]
    ns.dotplot(ax, labels, vals, colors=cols)
    ax.axvline(dm, color=ns.NOMINAL[4], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("RMSE, cell-line-disjoint")
    ax.set_xlim(0, 3.0)
    ax.set_yticklabels(labels, fontsize=6)
    ns.headroom(ax)          # keeps the panel letter inside, off the top rule
    ax.set_title("Against response statistics", fontweight="bold", pad=3)
    ax.text(dm + 0.08, 2.2, "per-drug mean,\nno cell features", fontsize=5.4,
            color=ns.NOMINAL[4], va="center", ha="left")


def panel_c(ax):
    """The ranking reverses between interpolation and cell-line-disjoint evaluation."""
    names = [b[0] for b in BLIND]
    x = np.arange(len(names))
    for vals, colour, mk, ls, lab in [
            ([b[1] for b in RANDOM], ns.NOMINAL[0], "o", "-", "random pairs"),
            ([b[1] for b in BLIND], ns.NOMINAL[1], "s", "--", "cell-line-disjoint")]:
        ax.plot(x, vals, color=colour, marker=mk, ls=ls, ms=3.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("RMSE")
    ns.headroom(ax)          # keeps the panel letter inside, off the title
    ax.set_title("Protocol changes the ordering", fontweight="bold", pad=3)
    ax.legend(loc="upper right", handlelength=1.8)


def panel_d(ax):
    """Seed-to-seed stability of the matched comparison."""
    t = pd.read_csv(os.path.join(ROOT, "source_data", "Table_S1_multiseed_internal.csv"))
    t = t[~t.seed.astype(str).isin(["mean", "sd"])]
    for i, (model, colour, mk) in enumerate([("BCGDRP", ns.NOMINAL[0], "o"),
                                             ("BANDRP-gate", ns.NOMINAL[1], "s")]):
        g = t[t.model == model]
        ax.plot(g.seed.astype(int), g.test_rmse, color=colour, marker=mk, ms=3.4,
                ls=["-", "--"][i], label=model)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Test RMSE")
    ax.set_xticks([2020, 2021, 2022, 2023, 2024])
    ax.set_xticklabels(["2020", "2021", "2022", "2023", "2024"], fontsize=6)
    ax.set_title("Stability across seeds", fontweight="bold", pad=3)
    ns.legend_clear(ax, handlelength=1.8)


def _cal(ax, centred, title):
    for (name, path), colour, mk in zip(CASES.items(), [ns.NOMINAL[0], ns.NOMINAL[1]], ["o", "s"]):
        lg = pd.read_csv(path)
        if centred:
            x = lg.pred_logIC50 - lg.groupby("drug_name").pred_logIC50.transform("mean")
            y = lg.true_logIC50 - lg.groupby("drug_name").true_logIC50.transform("mean")
        else:
            x, y = lg.pred_logIC50, lg.true_logIC50
        ax.scatter(x, y, s=0.6, alpha=0.10, color=colour, linewidths=0, rasterized=True)
        s = stats.linregress(x, y)
        xs = np.linspace(x.quantile(.005), x.quantile(.995), 50)
        ax.plot(xs, s.intercept + s.slope * xs, color=colour, lw=1.1, ls=["-", "--"][mk == "s"],
                label=f"{name}  slope {s.slope:.2f}")
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=ns.INK, lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("Predicted" + (" (drug-centred)" if centred else " ln(IC$_{50}$)"))
    ax.set_ylabel("Observed" + (" (drug-centred)" if centred else " ln(IC$_{50}$)"))
    ax.set_title(title, fontweight="bold", pad=3)
    ax.legend(loc="upper left", handlelength=1.6)


def panel_g(ax):
    """Rescaling the compressed component costs accuracy."""
    c = pd.read_csv(os.path.join(RES, "calibration_shrinkage.csv"))
    c = c[c.tissue.str.contains(r"\[", na=False)]
    labels, vals, cols = [], [], []
    for _, r in c.iterrows():
        tis, lab = r.tissue.split(" [")
        short = "match spread" if "spread" in lab else "MSE-optimal"
        labels.append(f"{tis.split('_')[0].title()[:7]}, {short}")
        vals.append(100 * (r.rmse_after - r.rmse_before) / r.rmse_before)
        cols.append(ns.NOMINAL[1] if vals[-1] > 0 else ns.NOMINAL[0])
    ns.dotplot(ax, labels, vals, colors=cols, vline=0.0)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_xlabel("Change in RMSE (%)")
    ax.set_title("Compression is not a fixable error", fontweight="bold", pad=3)


def panel_h(ax):
    """Per-drug shrinkage: almost every compound is compressed."""
    d = pd.read_csv(os.path.join(RES, "calibration_shrinkage_per_drug.csv"))
    for (tis, g), colour, ls in zip(d.groupby("tissue"), [ns.NOMINAL[1], ns.NOMINAL[0]], ["--", "-"]):
        v = np.clip(g.slope, -0.5, 2.0)
        ax.hist(v, bins=28, histtype="step", color=colour, lw=1.1, ls=ls,
                label=f"{tis.split('_')[0].title()} (n={len(g)})")
    ax.axvline(1.0, color=ns.INK, lw=0.7, ls=":")
    ax.text(1.05, ax.get_ylim()[1] * 0.55, "calibrated", fontsize=5.6, color=ns.INK, rotation=90)
    ax.set_xlabel("Drug-centred slope")
    ax.set_ylabel("Compounds")
    ax.set_title("Compression is near-universal", fontweight="bold", pad=3)
    ns.legend_clear(ax, handlelength=1.6)


def panel_i(ax):
    """What each predictor achieves on the four reported metrics, primary split."""
    p = pd.read_csv(os.path.join(RES, "naive_baselines_primary_split.csv")).set_index("predictor")
    metrics = [("RMSE", p.loc["NaiveDrugMeanPredictor", "RMSE"], 1.3644, True),
               ("$R^2$", p.loc["NaiveDrugMeanPredictor", "R2"], 0.7568, False),
               ("PCC", p.loc["NaiveDrugMeanPredictor", "PCC"], 0.8706, False),
               ("SCC", p.loc["NaiveDrugMeanPredictor", "SCC"], 0.8381, False)]
    x = np.arange(len(metrics))
    gains = [100 * (b - m) / b if lower else 100 * (m - b) / b
             for _, b, m, lower in metrics]
    ns.dotplot(ax, [m[0] for m in metrics], gains, colors=ns.NOMINAL[0], vline=0.0)
    ax.set_xlabel("Gain over the per-drug mean (%)")
    ax.set_title("Margin on each metric", fontweight="bold", pad=3)
    for yi, g in zip(np.arange(len(metrics))[::-1], gains):
        ax.text(g + 0.6, yi, f"{g:.1f}", fontsize=5.8, va="center", color=ns.INK)
    ax.set_xlim(0, max(gains) * 1.32)


def main():
    fig = ns.new_figure(ns.W_2COL, H_FIG)
    # Heights are stated in millimetres, not shares. The matrix resolves 169 rows,
    # so 70 mm gives it 0.41 mm per row and it stays the densest panel here at
    # 5.4 drawn elements per mm2. Everything else is a line or a scatter and needs
    # a fraction of that, so the rest of the page goes to more panels.
    outer = ns.grid_mm(fig, 3, 1, left=15, right=3, top=4, bottom=11,
                       hgap=6.0, rows_mm=[68.0, 5.0, 73.0])
    ax = {}
    ax["a"], legs, im = panels.draw(fig, outer[0])

    keys = fig.add_subplot(outer[1]); keys.axis("off")
    kp = keys.get_position()
    cax = fig.add_axes([kp.x0, kp.y0 + kp.height * 0.30, 0.072, kp.height * 0.42])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"ln($IC_{50}$)", fontsize=4.8, labelpad=0.8)
    cb.ax.tick_params(labelsize=4.4, length=1.2, pad=0.8)
    cb.outline.set_linewidth(0.4)
    ptop, pcol, ttop, tcol = legs
    from matplotlib.lines import Line2D
    def handles(names, colours):
        return [Line2D([], [], marker="s", ls="", ms=2.4, mfc=c, mec="none")
                for c in colours[:len(names)]]
    lg1 = keys.legend(handles(ptop, pcol), [panels.short_pathway(n) for n in ptop],
                      title="Target pathway", loc="upper left", bbox_to_anchor=(0.135, 1.55),
                      ncol=4, frameon=False, fontsize=4.5, title_fontsize=4.9,
                      handlelength=0.6, columnspacing=0.6, handletextpad=0.25,
                      borderpad=0.0, labelspacing=0.12)
    keys.add_artist(lg1)
    lg2 = keys.legend(handles(ttop, tcol), [panels.short_tissue(n) for n in ttop],
                      title="Tissue", loc="upper left", bbox_to_anchor=(0.60, 1.55), ncol=4,
                      frameon=False, fontsize=4.5, title_fontsize=4.9, handlelength=0.6,
                      columnspacing=0.6, handletextpad=0.25, borderpad=0.0, labelspacing=0.12)
    for lg in (lg1, lg2):
        lg.get_title().set_fontweight("bold")

    # eight satellites, two rows of four
    W, _ = ns.figure_mm(fig)
    span = W - 15 - 3
    gs = outer[2].subgridspec(2, 4, wspace=13.0 / ((span - 39.0) / 4),
                              hspace=14.0 / ((96.0 - 14.0) / 2))
    jobs = [panel_a, panel_b, panel_c, panel_d,
            lambda a: _cal(a, False, "Calibration, all pairs"),
            lambda a: _cal(a, True, "Calibration, drug-centred"),
            panel_h, panel_i]
    for i, (L, fn) in enumerate(zip("bcdefghi", jobs)):
        ax[L] = fig.add_subplot(gs[i // 4, i % 4]); fn(ax[L])

    for L, a in ax.items():
        ns.panel_label(fig, a, L)
    ns.assert_hero(ax["a"], min_mm2=9000.0)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(FIGS, "fig2.pdf"), ns.W_2COL, H_FIG)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
