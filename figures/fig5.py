#!/usr/bin/env python
"""Figure 5: module and modality ablations, and the component sweeps.

Every ablation here is a single training run on one split, so the panels
characterise component behaviour and hyperparameter sensitivity rather than
estimate effect sizes.  Differences are read against the response-statistics
baseline shown as a reference line, which is the scale on which they matter.

Run from the repository root.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import LinearSegmentedColormap

import style as ns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, "source_data")
RES = os.path.join(ROOT, "results", "analysis")
FIGS = os.path.join(ROOT, "outputs", "figures")
FULL_RMSE, FULL_PCC = 1.3644, 0.8706


def _baseline():
    p = pd.read_csv(os.path.join(RES, "naive_baselines_primary_split.csv")).set_index("predictor")
    return p.loc["NaiveDrugMeanPredictor", "RMSE"]


def ablation(ax, csv, title, wide=False):
    d = pd.read_csv(os.path.join(SD, csv))
    d = d.sort_values("RMSE", ascending=False)
    cols = [ns.NOMINAL[0] if m == "BCGDRP" else ns.NEUTRAL for m in d.Method]
    ns.dotplot(ax, d.Method.tolist(), d.RMSE.values, colors=cols,
               vline=min(d.RMSE.min(), FULL_RMSE) - (d.RMSE.max() - min(d.RMSE.min(), FULL_RMSE)) * 0.18)
    ax.axvline(FULL_RMSE, color=ns.NOMINAL[0], lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticklabels(d.Method.tolist(), fontsize=5.6)
    ax.set_xlabel("RMSE")
    lo, hi = min(d.RMSE.min(), FULL_RMSE), d.RMSE.max()
    pad = (hi - lo) * 0.18
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(title, fontweight="bold", pad=2.4, fontsize=6.0)


def sweep(ax, group, metric, xlabel, title, highlight=None):
    d = pd.read_csv(os.path.join(SD, "Fig3_component_sweeps.csv"))
    d = d[(d.panel_group == group) & (d.metric == metric)]
    for i, (v, g) in enumerate(d.groupby("variant")):
        g = g.sort_values("parameter_value")
        ax.plot(g.parameter_value, g.value, color=ns.NOMINAL[i % len(ns.NOMINAL)],
                marker=ns.MARKERS[i % len(ns.MARKERS)], ls=ns.LINESTYLES[i % len(ns.LINESTYLES)],
                ms=3.2, label=v.replace("_", " + "))
    if highlight is not None:
        ax.axvline(highlight, color=ns.INK, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight="bold", pad=2.4, fontsize=6.0)
    if d.variant.nunique() > 1:
        ns.legend_clear(ax, handlelength=1.6, fontsize=5.6, labelspacing=0.22)


def fusion(ax, metric, legend=True):
    d = pd.read_csv(os.path.join(SD, "Fig3_fusion_variants.csv"))
    d = d[d.metric == metric]
    names = {"exp_morgan_ban": "BAN", "exp_morgan_concat_mlp": "concat-MLP",
             "exp_morgan_cross_attention": "cross-attention",
             "meth_morgan_ban": "BAN", "meth_morgan_concat_mlp": "concat-MLP",
             "meth_morgan_cross_attention": "cross-attention"}
    exp = d[d.variant.str.startswith("exp")].sort_values("variant")
    meth = d[d.variant.str.startswith("meth")].sort_values("variant")
    lab = [names.get(v, v) for v in exp.variant]
    y = np.arange(len(lab))[::-1]
    ax.scatter(exp.value, y + 0.14, s=16, color=ns.NOMINAL[0], marker="o",
               linewidths=0, label="expression + Morgan", zorder=3)
    ax.scatter(meth.value, y - 0.14, s=16, color=ns.NOMINAL[1], marker="s",
               linewidths=0, label="methylation + Morgan", zorder=3)
    full = FULL_RMSE if metric == "RMSE" else FULL_PCC
    ax.axvline(full, color=ns.NOMINAL[4], lw=0.8, ls=(0, (3, 2)), zorder=1, label="BCGDRP")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=5.8)
    ax.set_xlabel(metric)
    ax.set_title(f"Fusion strategy, {metric}", fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_ylim(-0.6, len(lab) - 0.3)
    if legend:
        ns.legend_clear(ax, handlelength=1.1, fontsize=5.0,
                        labelspacing=0.18, borderpad=0.1)


def panel_scale(ax):
    """Every ablation effect, against what the model adds over response statistics."""
    base = _baseline()
    mod = pd.read_csv(os.path.join(SD, "Table_ablation_module.csv"))
    moda = pd.read_csv(os.path.join(SD, "Table_ablation_modal.csv"))
    gap = base - FULL_RMSE
    rows = [(m, (r - FULL_RMSE) / gap * 100)
            for m, r in zip(mod.Method, mod.RMSE) if m != "BCGDRP"]
    rows += [(m, (r - FULL_RMSE) / gap * 100) for m, r in zip(moda.Method, moda.RMSE)]
    rows.append(("BANDRP", (1.3771 - FULL_RMSE) / gap * 100))
    rows.sort(key=lambda t: t[1])
    ns.dotplot(ax, [r[0] for r in rows], [r[1] for r in rows],
               colors=ns.NOMINAL[3], vline=0.0)
    ax.axvline(100, color=ns.NOMINAL[1], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(101, len(rows) - 1.4, "per-drug\nmean", fontsize=5.4, color=ns.NOMINAL[1], va="center")
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.4)
    ax.set_xlabel("RMSE cost, % of model-baseline gap")
    ax.set_xlim(0, 118)
    ax.set_title("Ablations on a common scale", fontweight="bold", pad=2.4, fontsize=6.0)



def _metric_heat(ax, df, label_col, title, metrics=("RMSE", "PCC", "SCC", "R2")):
    """Variant against metric, every value printed.

    A lollipop of the same table shows one metric and throws the other three away.
    Colour is normalized within each column because the metrics do not share units,
    and each cell carries its own number so the panel is readable without the scale.
    """
    cols = [m for m in metrics if m in df.columns]
    A = df[cols].astype(float).values
    Z = np.zeros_like(A)
    for j, m in enumerate(cols):
        v = A[:, j]
        rng = v.max() - v.min()
        z = (v - v.min()) / rng if rng else np.zeros_like(v)
        Z[:, j] = 1.0 - z if m == "RMSE" else z      # better is always darker
    cmap = LinearSegmentedColormap.from_list("m", sorted(ns.LADDER, key=ns._luminance)[::-1])
    ax.imshow(Z, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([r"$R^2$" if c == "R2" else c for c in cols], fontsize=5.4)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df[label_col], fontsize=5.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=4.8,
                    color="white" if Z[i, j] > 0.62 else ns.INK)
    ax.set_title(title, fontweight="bold", pad=2.4, fontsize=6.0)
    ax.set_xlabel("Darker = better within each metric column",
                  fontsize=5.0, labelpad=2.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def ablation_heat(ax):
    a = pd.read_csv(os.path.join(SD, "Table_ablation_module.csv"))
    b = pd.read_csv(os.path.join(SD, "Table_ablation_modal.csv"))
    d = pd.concat([a, b], ignore_index=True)
    _metric_heat(ax, d, "Method", "Every ablation, every metric")


def fusion_heat(ax):
    f = pd.read_csv(os.path.join(SD, "Fig3_fusion_variants.csv"))
    w = f.pivot_table(index="variant", columns="metric", values="value").reset_index()
    w = w.sort_values("RMSE")
    # raw variant keys are code identifiers and too long for a three-across panel
    pretty = {"ban": "BAN", "concat_mlp": "concat-MLP", "cross_attention": "cross-att."}
    def name(v):
        lead = "exp" if v.startswith("exp") else "meth"
        for k, lab in pretty.items():
            if v.endswith(k):
                return f"{lab}, {lead}"
        return "BCGDRP (full)" if v.startswith("full") else v
    w["variant"] = [name(v) for v in w.variant]
    _metric_heat(ax, w, "variant", "Fusion strategies", metrics=("RMSE", "PCC"))



def sensitivity(ax):
    """How far each knob can move each metric, relative to the chosen setting.

    The six sweep panels each show one knob. None of them answers the question a
    reader actually has, which is whether any of these choices matters as much as
    the components do, so it is answered here on one axis.
    """
    d = pd.read_csv(os.path.join(SD, "Fig3_component_sweeps.csv"))
    chosen = {"dcg_alpha": 0.2, "lambda_ban": 0.005, "temperature": 0.12}
    nice = {"dcg_alpha": r"Gating $\alpha$", "lambda_ban": r"$\lambda_{\mathrm{ban}}$",
            "temperature": r"Temperature $\tau$"}
    rows = []
    for g, sub in d.groupby("panel_group"):
        for m, mm in sub.groupby("metric"):
            v = mm.groupby("parameter_value").value.mean()
            ref = v.reindex([chosen[g]]).iloc[0]
            if not np.isfinite(ref):
                continue
            pct = (v - ref) / abs(ref) * 100.0
            rows.append((f"{nice[g]}, {m}", pct.min(), pct.max(), m))
    rows.sort(key=lambda r: r[2] - r[1])
    y = np.arange(len(rows))
    for i, (lab, lo, hi, m) in enumerate(rows):
        colour = ns.NOMINAL[0] if m == "RMSE" else ns.NOMINAL[1]
        ax.plot([lo, hi], [i, i], color=colour, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([lo, hi], [i, i], s=9, color=colour, linewidths=0, zorder=4)
        ax.text(hi + 0.06, i, f"{hi - lo:.2f}", va="center", fontsize=4.6, color=ns.INK)
    ax.axvline(0, color=ns.INK, lw=0.6, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Change from the chosen setting (%), across the whole sweep")
    ax.set_title("How much each hyperparameter can move each metric",
                 fontweight="bold", pad=2.4, fontsize=6.0)
    from matplotlib.lines import Line2D
    ns.legend_clear(ax, handles=[Line2D([], [], color=ns.NOMINAL[0], lw=2.2, label="RMSE"),
                                 Line2D([], [], color=ns.NOMINAL[1], lw=2.2, label="PCC")],
                    fontsize=5.4, handlelength=1.2, ncol=2)


def main():
    fig = ns.new_figure(ns.W_2COL, ns.H_MAIN)
    # Every sweep on both reported metrics rather than one, which is eleven panels
    # of content that already existed and was being shown four at a time.
    gs = ns.grid_mm(fig, 4, 3, left=23, right=3, top=4, bottom=10,
                    wgap=15.0, hgap=12.0)
    ax = {}
    jobs = [
        ("a", lambda a: sweep(a, "dcg_alpha", "RMSE", r"$\alpha$", "Gating, RMSE", 0.2)),
        ("b", lambda a: sweep(a, "dcg_alpha", "PCC", r"$\alpha$", "Gating, PCC", 0.2)),
        ("c", lambda a: sweep(a, "lambda_ban", "RMSE", r"$\lambda_{\mathrm{ban}}$",
                              "Contrastive weight, RMSE", 0.005)),
        ("d", lambda a: sweep(a, "lambda_ban", "PCC", r"$\lambda_{\mathrm{ban}}$",
                              "Contrastive weight, PCC", 0.005)),
        ("e", lambda a: sweep(a, "temperature", "RMSE", r"$\tau$", "Temperature, RMSE", 0.12)),
        ("f", lambda a: sweep(a, "temperature", "PCC", r"$\tau$", "Temperature, PCC", 0.12)),
        ("g", ablation_heat),
        ("h", fusion_heat),
        ("i", panel_scale),
    ]
    for i, (L, fn) in enumerate(jobs):
        ax[L] = fig.add_subplot(gs[i // 3, i % 3]); fn(ax[L])
    # the fourth row carries the sensitivity summary across the full width
    ax["j"] = fig.add_subplot(gs[3, :]); sensitivity(ax["j"])
    for L, a in ax.items():
        ns.panel_label(fig, a, L)
    ns.assert_layout(fig, max_gap_mm=5.0, verbose=False)
    ns.assert_no_overlap(fig, verbose=True)
    ns.assert_aspect(fig, verbose=True)
    ns.assert_margins(fig, verbose=False)
    ns.assert_density(fig)
    ns.save(fig, os.path.join(FIGS, "fig5.pdf"), ns.W_2COL, ns.H_MAIN)
    plt.close(fig)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
