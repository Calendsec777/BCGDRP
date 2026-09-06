#!/usr/bin/env python
"""Build Figure 7: candidate nomination audit and CCK-8 validation summary.

Panels a and b show aggregate dose-response curves in 5637 and patient-derived
cells. Panels c--f audit the nomination, annotated pairs, potency and calibration.
Panel g summarizes the fitted parameters for all seven curves. Replicate-well
curves are shown at readable size in Supplementary Figure S4.

Authored at final size.  No resizebox, no tight bounding box, so source point
size equals rendered point size.

Run from the repository root after experiments/analysis/dose.py and baselines.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

import style as ns
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "analysis")
RES_EXTRA = RES
DEST = [os.path.join(ROOT, "outputs", "figures")]

MM = 1 / 25.4
# Height is a consequence of the panel widths, not a page-area target: an
# ordinary panel is at most as tall as it is wide. Moving the seven replicate
# curves to Supplementary Figure S4 leaves three content rows and removes the
# page-filling height that the earlier fourteen-panel version required.
W_MM, H_MM = 180.0, 146.0
INK = "#28303F"
# nominal palette: chroma fixed, lightness free, one pure neutral
C = {"Daporinad": ns.NOMINAL[2], "Dactolisib": ns.NOMINAL[4], "Dinaciclib": ns.NOMINAL[0]}
NEUTRAL = "#8C8C8C"
MK = {"Daporinad": "o", "Dactolisib": "s", "Dinaciclib": "^"}
LS = {"Daporinad": "-", "Dactolisib": "--", "Dinaciclib": "-."}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.0,
})


def ll4(x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + 10 ** ((x - log_ic50) * hill))


def panel_label(fig, ax, letter, pad=0.075):
    """Place the panel letter in figure coordinates, clear of wide tick labels."""
    b = ax.get_position()
    fig.text(max(b.x0 - pad, 0.006), min(b.y1 + 0.035, 0.995), letter,
             fontsize=8, fontweight="bold", va="top", ha="left", color=INK)


def draw_curves(ax, reps, fits, cells, title):
    sub = reps[reps.cell_line == cells]
    for drug in ["Daporinad", "Dactolisib", "Dinaciclib"]:
        g = sub[sub.drug == drug]
        if g.empty:
            continue
        g = g[g.concentration_uM > 0]
        f = fits[(fits.cell_line == cells) & (fits.drug == drug)].iloc[0]
        mu = g.groupby("concentration_uM").viability.agg(["mean", "std"]).reset_index()
        ax.scatter(np.log10(mu.concentration_uM), mu["mean"], s=15,
                   color=C[drug], marker=MK[drug], linewidths=0, zorder=3)
        ax.errorbar(np.log10(mu.concentration_uM), mu["mean"], yerr=mu["std"],
                    fmt="none", ecolor=C[drug], elinewidth=0.6, capsize=1.2, zorder=4)
        xs = np.linspace(np.log10(g.concentration_uM.min()) - 0.3,
                         np.log10(g.concentration_uM.max()) + 0.3, 300)
        ax.plot(xs, ll4(xs, f.bottom, f.top, np.log10(f.ic50_uM), f.hill) * 100,
                color=C[drug], ls=LS[drug], lw=1.0, zorder=5,
                label=f"{drug}  {f.ic50_nM:.3g} nM")
    ax.axhline(50, color=NEUTRAL, lw=0.5, ls=":", zorder=1)
    ax.set_xlabel("Concentration, log$_{10}$ " + r"$\mu$M")
    ax.set_ylabel("Relative viability (%)")
    ax.set_title(title, fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_ylim(-6, 118)
    ns.legend_clear(ax, handlelength=1.6, borderpad=0.1,
              labelspacing=0.25, handletextpad=0.5)



def fit_parameters(ax):
    """Summarize the parameters defining all seven fitted curves."""
    f = pd.read_csv(os.path.join(ROOT, "source_data", "Fig5_wetlab_ic50_fits.csv"))
    system_rank = {
        "5637": 0,
        "MDA-MB-231": 1,
        "patient-derived bladder cancer cells": 2,
    }
    drug_rank = {"Daporinad": 0, "Dactolisib": 1, "Dinaciclib": 2}
    f = (f.assign(_system=f.cell_line.map(system_rank), _drug=f.drug.map(drug_rank))
           .sort_values(["_system", "_drug"])
           .drop(columns=["_system", "_drug"]))
    lab = [("Patient" if str(c).startswith("patient") else str(c)) + f"\n{d}"
           for c, d in zip(f.cell_line, f.drug)]
    cols = ["ic50_nM", "hill", "r2", "top", "bottom"]
    names = [r"$IC_{50}$ (nM)", "Hill", r"$R^2$", "Top", "Bottom"]
    A = f[cols].astype(float).values.copy()
    A[:, 0] = np.log10(A[:, 0])
    Z = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j]; rng = v.max() - v.min()
        z = (v - v.min()) / rng if rng else np.zeros_like(v)
        # A larger IC50 is a less potent compound, so the potency column is
        # inverted to keep dark meaning potent, which is what dark means in the
        # response landscapes. The remaining columns describe the curve shape and
        # carry no better-or-worse direction.
        Z[:, j] = 1.0 - z if cols[j] == "ic50_nM" else z
    cmap = LinearSegmentedColormap.from_list("m", sorted(ns.LADDER, key=ns._luminance)[::-1])
    ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    raw = f[cols].astype(float).values
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            txt = f"{raw[i, j]:.3g}" if j == 0 else f"{raw[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=4.5,
                    color="white" if Z[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=4.6)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=4.4)
    ax.set_title("Fitted parameters", fontweight="bold", pad=2.4, fontsize=6.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def main():
    reps = pd.read_csv(os.path.join(ROOT, "source_data", "Fig5_wetlab_replicates.csv"))
    fits = pd.read_csv(os.path.join(ROOT, "source_data", "Fig5_wetlab_ic50_fits.csv"))
    audit = pd.read_csv(os.path.join(RES, "nomination_baseline_audit_5637.csv"))

    fig = ns.new_figure(W_MM, H_MM)
    # The two dose-response panels share both axes, so they sit 5 mm apart and read
    # as one experiment across two cellular systems. Four audit panels form the
    # second row; the fitted-parameter matrix is centred below them. Replicate-well
    # curves are moved, rather than copied, to Supplementary Figure S4.
    outer = ns.grid_mm(fig, 3, 1, left=16, right=3, top=5, bottom=9,
                       hgap=10.0, rows_mm=[44.0, 30.0, 38.0])
    W0, _ = ns.figure_mm(fig)
    span = W0 - 16 - 3
    top = outer[0].subgridspec(1, 2, wspace=5.0 / ((span - 5.0) / 2))
    ax_b = fig.add_subplot(top[0, 0])
    ax_c = fig.add_subplot(top[0, 1], sharex=ax_b, sharey=ax_b)
    gs = outer[1].subgridspec(1, 4, wspace=13.0 / ((span - 39.0) / 4))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[0, 3])
    parameter_row = outer[2].subgridspec(
        1, 3, width_ratios=[0.22, 0.56, 0.22], wspace=0.0
    )
    ax_g = fig.add_subplot(parameter_row[0, 1])

    # ---- a: nomination audit -------------------------------------------------
    a = audit.sort_values("mean_lnIC50_visible_cells").reset_index(drop=True)
    y = np.arange(len(a))[::-1]
    ax_a.hlines(y, 0, a.mean_lnIC50_visible_cells, color=NEUTRAL, lw=0.7, zorder=2)
    col = [C["Daporinad"] if d == "Daporinad" else NEUTRAL for d in a.drug]
    ax_a.scatter(a.mean_lnIC50_visible_cells, y, s=13, color=col, zorder=3,
                 linewidths=0)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(a.drug, fontsize=5.6)
    ax_a.axvline(0, color=INK, lw=0.5, zorder=1)
    # "over model-visible cells" lives in the caption. At 30 mm of panel width a
    # 60 mm label runs under the neighbour, which is what it was doing.
    ax_a.set_xlabel(r"Mean observed ln($IC_{50}$ [$\mu$M])")
    ax_a.set_title("5637 candidates", fontweight="bold", pad=2.4, fontsize=6.0)
    # Every other candidate has a positive mean, so the whole region left of the zero
    # line is empty. Put the label there rather than over the markers.
    ax_a.annotate("no-model rank 1\nBCGDRP rank 2",
                  xy=(a.mean_lnIC50_visible_cells[0], y[0]),
                  xytext=(0.03, 0.22), textcoords="axes fraction",
                  fontsize=5.8, color=C["Daporinad"], ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", color=C["Daporinad"], lw=0.6,
                                  shrinkA=1.0, shrinkB=2.0))

    # ---- b, c: dose-response -------------------------------------------------
    draw_curves(ax_b, reps, fits, "5637", "5637 bladder cancer line")
    draw_curves(ax_c, reps, fits, "patient-derived bladder cancer cells",
                "Patient-derived bladder cells")
    # the pair shares both axes, so the second copy of the scale is redundant and
    # is what a 5 mm gap between them buys
    ax_c.tick_params(labelleft=False)
    ax_c.set_ylabel("")

    # ---- d: calibration ------------------------------------------------------
    # The predicted-against-recorded panel moves to the supplement; the
    # figure keeps the two cellular systems as its hero instead.
    if ax_d is not None:
        known = [("5637", "Dinaciclib", 150.2, 62.7),
                 ("MDA-MB-231", "Dinaciclib", 35.0, 292.8),
                 ("5637", "Dactolisib", 269.5, 384.7)]
        xs = np.arange(len(known))
        for i, (cl, dg, gd, pr) in enumerate(known):
            assay = fits[(fits.cell_line == cl) & (fits.drug == dg)].ic50_nM.iloc[0]
            lo = fits[(fits.cell_line == cl) & (fits.drug == dg)].ic50_ci_low_uM.iloc[0] * 1000
            hi = fits[(fits.cell_line == cl) & (fits.drug == dg)].ic50_ci_high_uM.iloc[0] * 1000
            ax_d.plot([i - 0.22, i - 0.22], [lo, hi], color=INK, lw=0.8, zorder=2)
            ax_d.scatter(i - 0.22, assay, s=20, color=INK, marker="o", zorder=3,
                         label="CCK-8 assay" if i == 0 else None, linewidths=0)
            ax_d.scatter(i, gd, s=20, color=C["Daporinad"], marker="s", zorder=3,
                         label="GDSC2 record" if i == 0 else None, linewidths=0)
            ax_d.scatter(i + 0.22, pr, s=20, color=C["Dactolisib"], marker="^", zorder=3,
                         label="BCGDRP prediction" if i == 0 else None, linewidths=0)
        ax_d.set_yscale("log")
        ax_d.set_ylim(3.5, 1500)
        ax_d.set_xticks(xs)
        ax_d.set_xticklabels([f"{cl}\n{dg}" for cl, dg, _, _ in known], fontsize=6)
        ax_d.set_xlim(-0.55, len(known) - 0.45)
        ax_d.set_ylabel(r"$IC_{50}$ (nM)")
        ax_d.set_title("Annotated pairs", fontweight="bold", pad=2.4, fontsize=6.0)
        ax_d.legend(loc="upper left", handlelength=1.0, borderpad=0.1,
                    labelspacing=0.25, handletextpad=0.4)

    # ---- e: potency across the three systems ---------------------------------
    order = ["Daporinad", "Dactolisib", "Dinaciclib"]
    systems = [("5637", "o"), ("patient-derived bladder cancer cells", "s"),
               ("MDA-MB-231", "^")]
    xs = np.arange(len(order))
    for (cl, mk), off in zip(systems, (-0.20, 0.0, 0.20)):
        v, x = [], []
        for i, dg in enumerate(order):
            r = fits[(fits.cell_line == cl) & (fits.drug == dg)]
            if len(r):
                v.append(r.ic50_nM.iloc[0]); x.append(i + off)
        lab = "Patient-derived" if cl.startswith("patient") else cl
        ax_e.scatter(x, v, s=22, marker=mk, linewidths=0, zorder=3,
                     color=[C[order[int(round(xi - off))]] for xi in x])
        ax_e.scatter([], [], s=22, marker=mk, linewidths=0, color=INK, label=lab)
    ax_e.set_yscale("log")
    ax_e.set_xticks(xs); ax_e.set_xticklabels(order, fontsize=6)
    ax_e.set_xlim(-0.5, len(order) - 0.5)
    ax_e.set_ylabel(r"$IC_{50}$ (nM)")
    ax_e.set_title("Potency across systems", fontweight="bold", pad=2.4, fontsize=6.0)
    ax_e.set_ylim(0.15, 700)
    ax_e.legend(loc="upper left", handlelength=1.0, labelspacing=0.25)

    # ---- f: why absolute potency is not the usable output --------------------
    cal = pd.read_csv(os.path.join(RES_EXTRA, "calibration_shrinkage.csv"))
    cal = cal[cal.tissue.isin(["URINARY_TRACT", "BREAST"])]
    labels = ["All pairs\n(urinary)", "Drug-centred\n(urinary)",
              "All pairs\n(breast)", "Drug-centred\n(breast)"]
    vals = [cal.iloc[0].slope_overall, cal.iloc[0].slope_drug_centred,
            cal.iloc[1].slope_overall, cal.iloc[1].slope_drug_centred]
    cols = [NEUTRAL, C["Daporinad"], NEUTRAL, C["Daporinad"]]
    y = np.arange(len(labels))[::-1]
    ax_f.hlines(y, 0, vals, color=NEUTRAL, lw=0.8, zorder=2)
    ax_f.scatter(vals, y, s=18, color=cols, zorder=3, linewidths=0)
    ax_f.axvline(1.0, color=INK, lw=0.7, ls=":", zorder=1)
    ax_f.text(1.02, 1.6, "calibrated", fontsize=5.6, color=INK, rotation=90, va="center")
    ax_f.set_yticks(y); ax_f.set_yticklabels(labels, fontsize=5.6)
    ax_f.set_xlim(0, 1.35)
    ax_f.set_xlabel("Calibration slope")
    ax_f.set_title("Potency is compressed", fontweight="bold", pad=2.4, fontsize=6.0)

    fit_parameters(ax_g)
    order = [(ax_b, "a"), (ax_c, "b"), (ax_a, "c"), (ax_d, "d"), (ax_e, "e"), (ax_f, "f")]
    order += [(ax_g, "g")]
    for ax, L in order:
        ns.panel_label(fig, ax, L)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(DEST[0], "fig7.pdf"), W_MM, H_MM)
    plt.close(fig)


if __name__ == "__main__":
    main()
