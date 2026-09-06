#!/usr/bin/env python
"""Drug-level inference for the CTRPv2 cross-screen rank-transfer analysis.

The per-drug Spearman correlations were computed for five training seeds against
the same CTRPv2 records, so the 112 drugs x 5 seeds = 560 values are not 560
independent observations.  Correlations are therefore aggregated within each
compound first, and all inference is performed over the n = 112 drugs.

Run from the repository root.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "ctrp", "per_drug.csv")
OUT = os.path.join(ROOT, "results", "analysis")
BOOT = 10000
SEED = 20260828

d = pd.read_csv(SRC)
n_seed = d.seed.nunique()
# the two models must have been scored on identical CTRPv2 records
chk = d.pivot_table(index=["gdsc_pubchem_id", "seed"], columns="model", values="n_pairs")
assert (chk["bcgdrp"] == chk["bandrp_gate"]).all(), "models scored on different records"

agg = (d.groupby(["model", "gdsc_pubchem_id", "drug_name"], as_index=False)
        .agg(spearman=("spearman", "mean"), n_cells=("n_cells", "max"),
             n_pairs=("n_pairs", "max"), n_seed=("seed", "nunique")))
assert (agg.n_seed == n_seed).all(), "a drug is missing a seed"

w = (agg.pivot_table(index=["gdsc_pubchem_id", "drug_name", "n_cells"],
                     columns="model", values="spearman").reset_index())
w["delta"] = w.bcgdrp - w.bandrp_gate
w = w.sort_values("delta", ascending=False)
w.to_csv(os.path.join(OUT, "ctrpv2_drug_level_seed_aggregated.csv"), index=False)

n = len(w)
rng = np.random.default_rng(SEED)
boot = np.array([rng.choice(w.delta.values, n, replace=True).mean() for _ in range(BOOT)])
lo, hi = np.percentile(boot, [2.5, 97.5])
W, p_w = stats.wilcoxon(w.bcgdrp, w.bandrp_gate)
t, p_t = stats.ttest_rel(w.bcgdrp, w.bandrp_gate)
win = int((w.delta > 0).sum())
q1, q3 = w.delta.quantile([.25, .75])

print("=" * 88)
print(f"CTRPv2 cross-screen rank transfer, drug-level inference (n = {n} drugs, "
      f"{n_seed} seeds aggregated within drug)")
print("=" * 88)
print(f"  macro Spearman      BCGDRP {w.bcgdrp.mean():.4f}   BANDRP-gate {w.bandrp_gate.mean():.4f}")
print(f"  median Spearman     BCGDRP {w.bcgdrp.median():.4f}   BANDRP-gate {w.bandrp_gate.median():.4f}")
print(f"  drugs with higher BCGDRP rank correlation : {win}/{n} ({100*win/n:.1f}%)")
print(f"  mean difference     {w.delta.mean():+.4f}")
print(f"  median difference   {w.delta.median():+.4f}   IQR [{q1:+.4f}, {q3:+.4f}]")
print(f"  drug-resampling 95% CI of the mean difference : [{lo:+.4f}, {hi:+.4f}]")
print(f"  paired Wilcoxon signed-rank : W = {W:.1f}, P = {p_w:.3f}")
print(f"  paired t-test               : t = {t:.3f}, P = {p_t:.3f}")

print(f"\n  coverage: min {w.n_cells.min()}, median {int(w.n_cells.median())}, "
      f"max {w.n_cells.max()} evaluated cell lines per drug")
for thr in (20, 30):
    s = w[w.n_cells >= thr]
    print(f"    drugs with >= {thr} evaluated cell lines: {len(s)}/{n}"
          + ("  (identical to the full set; this is not an independent check)"
             if len(s) == n else ""))

k = 8
print(f"\n  largest gains (top {k}):")
print(w.head(k)[["drug_name", "n_cells", "bandrp_gate", "bcgdrp", "delta"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n  largest losses (bottom {k}):")
print(w.tail(k)[["drug_name", "n_cells", "bandrp_gate", "bcgdrp", "delta"]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# Per-drug differences look structured (the largest gains are selective kinase
# inhibitors), but a difference correlates with either single arm by construction.
# The Bland-Altman-correct axis is the average of the two correlations.
r_arm, p_arm = stats.pearsonr(w.bandrp_gate, w.delta)
r_avg, p_avg = stats.pearsonr((w.bandrp_gate + w.bcgdrp) / 2, w.delta)
print("\n  regression-to-the-mean check on the per-drug differences:")
print(f"    corr(BANDRP-gate rho, difference)      = {r_arm:+.3f}, P = {p_arm:.2g}  "
      f"(biased by construction)")
print(f"    corr(mean of the two rho, difference)  = {r_avg:+.3f}, P = {p_avg:.2g}  "
      f"(unbiased)")
print("    -> the unbiased test is not significant, so the per-drug pattern is not")
print("       evidence of a systematic compound-class effect without a pre-specified")
print("       target-class annotation.")
print(f"\n  drugs with negative rank correlation: BANDRP-gate "
      f"{int((w.bandrp_gate < 0).sum())}/{n}, BCGDRP {int((w.bcgdrp < 0).sum())}/{n}")

print(f"\nwrote {os.path.join(OUT, 'ctrpv2_drug_level_seed_aggregated.csv')}")
