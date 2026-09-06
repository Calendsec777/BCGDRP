#!/usr/bin/env python
"""Supplementary figures S1-S4.

These carry the per-stratum detail behind the main figures: every compound, every
held-out cell line, every training seed and every dose-response well.  Heights are
free up to 225 mm here, unlike the main set.

Run from the repository root, after the analysis scripts.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as ns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "source_data")
RES = os.path.join(ROOT, "results", "analysis")
FIGS = os.path.join(ROOT, "outputs", "figures")


def _finalise(fig):
    ns.raise_text_floor(fig, min_pt=5.0)
    ns.assert_text_floor(fig, min_pt=5.0)


def s1_per_drug():
    """Split 63 and 154 named compounds over two readable vector pages."""
    cross = pd.read_csv(os.path.join(RES, "model_vs_ceiling_per_drug.csv"))
    cross = cross.sort_values("ceiling_rho")
    held = pd.read_csv(os.path.join(RES, "prioritisation_mode2_per_drug.csv"))
    held = held.pivot_table(
        index="drug_name", columns="tissue", values="scc_bcgdrp"
    ).dropna().sort_values("URINARY_TRACT")

    page1 = ns.new_figure(ns.W_2COL, 175.0)
    ax = page1.add_axes([0.205, 0.070, 0.775, 0.875])
    y = np.arange(len(cross))
    ax.hlines(y, cross.bcgdrp, cross.ceiling_rho, color=ns.NEUTRAL,
              lw=0.65, zorder=2)
    ax.scatter(cross.ceiling_rho, y, s=8, color=ns.NEUTRAL, marker="o",
               linewidths=0, label="two screens", zorder=3)
    ax.scatter(cross.bcgdrp, y, s=8, color=ns.NOMINAL[3], marker="s",
               linewidths=0, label="BCGDRP", zorder=4)
    ax.scatter(cross.bandrp_gate, y, s=8, color=ns.NOMINAL[0], marker="^",
               linewidths=0, label="BANDRP-gate", zorder=4)
    ax.axvline(0, color=ns.INK, lw=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(cross.drug_name, fontsize=5.0)
    ax.set_ylim(-1, len(cross))
    ax.set_xlabel(r"Within-drug Spearman $\rho$")
    ax.set_title("Cross-screen transfer, all 63 compounds with a measurable ceiling",
                 fontsize=7.0, fontweight="bold", pad=3)
    ax.legend(loc="lower right", ncol=3, fontsize=5.8, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.35)
    ns.panel_label(page1, ax, "a")
    _finalise(page1)

    page2 = ns.new_figure(ns.W_2COL, 225.0)
    gs = page2.add_gridspec(
        1, 2, left=0.155, right=0.990, bottom=0.050, top=0.955,
        wspace=0.56, width_ratios=[1, 1],
    )
    lo = float(min(held.URINARY_TRACT.min(), held.BREAST.min()))
    hi = float(max(held.URINARY_TRACT.max(), held.BREAST.max()))
    pad = (hi - lo) * 0.04
    split = (len(held) + 1) // 2
    blocks = (held.iloc[:split], held.iloc[split:])
    for j, block in enumerate(blocks):
        axis = page2.add_subplot(gs[0, j])
        yy = np.arange(len(block))
        axis.hlines(yy, block.URINARY_TRACT, block.BREAST,
                    color=ns.NEUTRAL, lw=0.55, zorder=2)
        axis.scatter(block.URINARY_TRACT, yy, s=7, color=ns.NOMINAL[3],
                     marker="o", linewidths=0, label="Urinary tract", zorder=3)
        axis.scatter(block.BREAST, yy, s=7, color=ns.NOMINAL[0],
                     marker="s", linewidths=0, label="Breast", zorder=3)
        axis.axvline(0, color=ns.INK, lw=0.5)
        axis.set_yticks(yy)
        axis.set_yticklabels(block.index, fontsize=5.0)
        axis.set_ylim(-1, len(block))
        axis.set_xlim(lo - pad, hi + pad)
        axis.set_xlabel(r"Within-drug Spearman $\rho$")
        axis.set_title(
            f"Held-out tissue transfer, compounds {j * len(blocks[0]) + 1}--"
            f"{j * len(blocks[0]) + len(block)}"
            f"{' (continued)' if j else ''}",
            fontsize=6.5, fontweight="bold", pad=3,
        )
        if j == 0:
            axis.legend(loc="lower right", fontsize=5.8, handlelength=1.0)
            ns.panel_label(page2, axis, "b")
    _finalise(page2)

    output = os.path.join(FIGS, "fig_s1.pdf")
    os.makedirs(FIGS, exist_ok=True)
    with PdfPages(output) as pdf:
        for page in (page1, page2):
            with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
                pdf.savefig(page)
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        page1.savefig(os.path.join(FIGS, "fig_s1.png"), dpi=450)
    print("  wrote fig_s1.pdf  2 pages: 180 x 175 mm; 180 x 225 mm")
    plt.close(page1)
    plt.close(page2)


def s2_per_cell():
    """Stack the 15- and 42-row tissues with equal physical row pitch."""
    data = pd.read_csv(os.path.join(RES, "within_cell_scc_disease_holdouts.csv"))
    fig = ns.new_figure(ns.W_2COL, 175.0)
    gs = fig.add_gridspec(
        2, 1, left=0.175, right=0.985, bottom=0.075, top=0.945,
        hspace=0.23, height_ratios=[15, 42],
    )
    lo = float(min(data.scc_bcgdrp.min(), data.scc_drug_mean.min()))
    hi = float(max(data.scc_bcgdrp.max(), data.scc_drug_mean.max()))
    pad = (hi - lo) * 0.04
    axes = []
    for j, (tissue, label, colour) in enumerate(
        (("URINARY_TRACT", "Urinary tract", ns.NOMINAL[3]),
         ("BREAST", "Breast", ns.NOMINAL[0]))
    ):
        ax = fig.add_subplot(gs[j, 0], sharex=axes[0] if axes else None)
        group = data[data.tissue == tissue].sort_values("scc_bcgdrp")
        y = np.arange(len(group))
        ax.hlines(y, group.scc_drug_mean, group.scc_bcgdrp,
                  color=ns.NEUTRAL, lw=0.65, zorder=2)
        ax.scatter(group.scc_drug_mean, y, s=10, color=ns.NEUTRAL,
                   marker="o", linewidths=0, label="per-drug mean", zorder=3)
        ax.scatter(group.scc_bcgdrp, y, s=11, color=colour,
                   marker="s", linewidths=0, label="BCGDRP", zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(group.cell_line_name, fontsize=5.0)
        ax.set_ylim(-1, len(group))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(r"Within-cell-line Spearman $\rho$")
        ax.set_title(f"{label} ({len(group)} held-out cell lines)",
                     fontsize=6.8, fontweight="bold", pad=3)
        if j == 0:
            ax.legend(loc="lower right", ncol=2, fontsize=5.8,
                      handlelength=1.0, columnspacing=0.9)
        ns.panel_label(fig, ax, "ab"[j])
        axes.append(ax)
    _finalise(fig)
    ns.save(fig, os.path.join(FIGS, "fig_s2.pdf"),
            ns.W_2COL, 175.0)
    plt.close(fig)


def s3_seeds():
    """Use categorical seed dumbbells rather than an implied trajectory."""
    data = pd.read_csv(os.path.join(SD, "Table_S1_multiseed_internal.csv"))
    data = data[~data.seed.astype(str).isin(["mean", "sd"])].copy()
    data["seed"] = data.seed.astype(int)
    fig = ns.new_figure(ns.W_2COL, 76.0)
    gs = fig.add_gridspec(1, 2, left=0.105, right=0.985, bottom=0.165,
                          top=0.865, wspace=0.36)
    specs = (("test_rmse", "RMSE", "Test RMSE"),
             ("test_scc", "Spearman correlation", r"Test Spearman $\rho$"))
    for j, (metric, title, xlabel) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j])
        pivot = data.pivot(index="seed", columns="model", values=metric).sort_index()
        y = np.arange(len(pivot))[::-1] + 1.5
        ax.hlines(y, pivot["BCGDRP"], pivot["BANDRP-gate"],
                  color=ns.NEUTRAL, lw=0.75, zorder=2)
        ax.scatter(pivot["BCGDRP"], y, s=18, color=ns.NOMINAL[3],
                   marker="o", linewidths=0, label="BCGDRP", zorder=4)
        ax.scatter(pivot["BANDRP-gate"], y, s=18, color=ns.NOMINAL[0],
                   marker="s", linewidths=0, label="BANDRP-gate", zorder=4)
        means = pivot.mean(axis=0)
        sds = pivot.std(axis=0, ddof=1)
        summary_y = 0.0
        ax.hlines(summary_y, means["BCGDRP"], means["BANDRP-gate"],
                  color=ns.NEUTRAL, lw=0.9, zorder=2)
        for model, colour, marker in (
            ("BCGDRP", ns.NOMINAL[3], "o"),
            ("BANDRP-gate", ns.NOMINAL[0], "s"),
        ):
            ax.errorbar(means[model], summary_y, xerr=sds[model], fmt=marker,
                        ms=4.2, color=colour, ecolor=colour, elinewidth=0.8,
                        capsize=1.5, zorder=5)
        xmin = min(float(pivot.min().min()), float((means - sds).min()))
        xmax = max(float(pivot.max().max()), float((means + sds).max()))
        ax.hlines(0.75, xmin, xmax, color="#D8DEE4", lw=0.6)
        ax.set_yticks(list(y) + [summary_y])
        ax.set_yticklabels([str(seed) for seed in pivot.index] + ["Mean ± SD"],
                           fontsize=5.6)
        ax.set_ylabel("Training seed")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7.0, fontweight="bold", pad=3)
        if j == 0:
            ax.legend(loc="best", fontsize=5.8, handlelength=1.0)
        ns.panel_label(fig, ax, "ab"[j])
    _finalise(fig)
    ns.save(fig, os.path.join(FIGS, "fig_s3.pdf"),
            ns.W_2COL, 76.0)
    plt.close(fig)


def s4_curves():
    """Every dose-response curve at replicate-well level."""
    reps = pd.read_csv(os.path.join(ROOT, "source_data", "Fig5_wetlab_replicates.csv"))
    fits = pd.read_csv(os.path.join(ROOT, "source_data", "Fig5_wetlab_ic50_fits.csv"))
    system_rank = {
        "5637": 0,
        "MDA-MB-231": 1,
        "patient-derived bladder cancer cells": 2,
    }
    drug_rank = {"Daporinad": 0, "Dactolisib": 1, "Dinaciclib": 2}
    fits = (fits.assign(_system=fits.cell_line.map(system_rank), _drug=fits.drug.map(drug_rank))
                .sort_values(["_system", "_drug"])
                .drop(columns=["_system", "_drug"]))
    combos = fits[["cell_line", "drug"]].values.tolist()
    h = 96.0
    fig = ns.new_figure(ns.W_2COL, h)
    # Four panels above and three centred below keep all seven curves at the same
    # physical size without leaving two empty cells in a 3 x 3 grid.
    panel_w, panel_h = 36.0, 34.0
    positions = [(10.5 + 41.0 * i, 56.0) for i in range(4)]
    positions += [(31.0 + 41.0 * i, 10.0) for i in range(3)]
    colours = {"Daporinad": ns.NOMINAL[2], "Dactolisib": ns.NOMINAL[4],
               "Dinaciclib": ns.NOMINAL[0]}
    markers = {"Daporinad": "o", "Dactolisib": "s", "Dinaciclib": "^"}
    linestyles = {"Daporinad": "-", "Dactolisib": "--", "Dinaciclib": "-."}
    axes = []
    for i, (cl, dg) in enumerate(combos):
        left, bottom = positions[i]
        ax = fig.add_axes(
            [left / ns.W_2COL, bottom / h, panel_w / ns.W_2COL, panel_h / h],
            sharey=axes[0] if axes else None,
        )
        g = reps[(reps.cell_line == cl) & (reps.drug == dg) & (reps.concentration_uM > 0)]
        f = fits[(fits.cell_line == cl) & (fits.drug == dg)].iloc[0]
        colour = colours[dg]
        ax.scatter(np.log10(g.concentration_uM), g.viability, s=5, alpha=0.7,
                   color=colour, marker=markers[dg], linewidths=0, zorder=3)
        xs = np.linspace(np.log10(g.concentration_uM.min()) - 0.3,
                         np.log10(g.concentration_uM.max()) + 0.3, 300)
        yy = f.bottom + (f.top - f.bottom) / (1 + 10 ** ((xs - np.log10(f.ic50_uM)) * f.hill))
        ax.plot(xs, yy * 100, color=colour, ls=linestyles[dg], lw=1.0, zorder=4)
        ax.axhline(50, color=ns.NEUTRAL, lw=0.5, ls=":", zorder=1)
        ax.axvline(np.log10(f.ic50_uM), color=ns.INK, lw=0.6, ls=(0, (2, 2)), zorder=2)
        short = "Patient-derived" if cl.startswith("patient") else cl
        ax.set_title(f"{short}, {dg}", fontweight="bold", pad=2.4, fontsize=6.0)
        ax.text(0.97, 0.94,
                f"$IC_{{50}}$ {f.ic50_nM:.3g} nM\n$R^2$ {f.r2:.3f}; Hill {f.hill:.2f}",
                transform=ax.transAxes, fontsize=5.0, va="top", ha="right", color=ns.INK)
        ax.set_ylim(-6, 118)
        ax.set_xlabel(r"Concentration, log$_{10}$ $\mu$M", labelpad=1.0)
        if i in (0, 4):
            ax.set_ylabel("Relative viability (%)", labelpad=1.0)
        else:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
        ns.panel_label(fig, ax, "abcdefg"[i])
        axes.append(ax)
    _finalise(fig)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig, verbose=False)
    ns.save(fig, os.path.join(FIGS, "fig_s4.pdf"), ns.W_2COL, h)
    plt.close(fig)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    s1_per_drug()
    s2_per_cell()
    s3_seeds()
    s4_curves()


if __name__ == "__main__":
    main()
