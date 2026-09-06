#!/usr/bin/env python3
"""Refit the released CCK-8 replicate measurements with a four-parameter curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "source_data" / "Fig5_wetlab_replicates.csv"
OUTPUT = ROOT / "results" / "analysis"
SEED = 20260828
BOOTSTRAPS = 2000


def logistic(x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + 10 ** ((x - log_ic50) * hill))


def fit_curve(frame: pd.DataFrame) -> dict:
    frame = frame[frame.concentration_uM > 0].copy()
    x = np.log10(frame.concentration_uM.to_numpy(float))
    y = frame.viability.to_numpy(float) / 100.0
    initial = [max(float(y.min()), 0.0), max(float(y.max()), 1.0), float(np.median(x)), 1.0]
    lower = [-0.1, 0.5, x.min() - 2, 0.3]
    upper = [0.6, 3.0, x.max() + 2, 15.0]
    estimate, _ = curve_fit(logistic, x, y, p0=initial, bounds=(lower, upper), maxfev=400000)
    residual = y - logistic(x, *estimate)
    r2 = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)

    generator = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAPS):
        index = generator.integers(0, len(x), len(x))
        try:
            sampled, _ = curve_fit(
                logistic,
                x[index],
                y[index],
                p0=estimate,
                bounds=(lower, upper),
                maxfev=80000,
            )
            draws.append(sampled[2])
        except (RuntimeError, ValueError):
            continue
    interval = np.percentile(draws, [2.5, 97.5]) if len(draws) > BOOTSTRAPS // 2 else [np.nan, np.nan]
    concentrations = frame.concentration_uM.to_numpy(float)
    return {
        "ic50_uM": float(10 ** estimate[2]),
        "r2": float(r2),
        "hill": float(estimate[3]),
        "bottom": float(estimate[0]),
        "top": float(estimate[1]),
        "ic50_ci_low_uM": float(10 ** interval[0]),
        "ic50_ci_high_uM": float(10 ** interval[1]),
        "ic50_within_tested_range": bool(x.min() <= estimate[2] <= x.max()),
        "conc_min_uM": float(concentrations.min()),
        "conc_max_uM": float(concentrations.max()),
        "n_concentrations": int(frame.concentration_uM.nunique()),
        "n_wells": int(len(frame)),
    }


def main() -> None:
    replicates = pd.read_csv(INPUT)
    required = {"source", "cell_line", "drug", "concentration_uM", "replicate", "viability"}
    missing = required - set(replicates.columns)
    if missing:
        raise ValueError(f"Replicate table is missing columns: {sorted(missing)}")

    rows = []
    for keys, frame in replicates.groupby(["source", "cell_line", "drug"], sort=False):
        source, cell_line, drug = keys
        rows.append({"source": source, "cell_line": cell_line, "drug": drug, **fit_curve(frame)})
    fits = pd.DataFrame(rows)
    fits["ic50_nM"] = fits.ic50_uM * 1000
    OUTPUT.mkdir(parents=True, exist_ok=True)
    replicates.to_csv(OUTPUT / "wetlab_replicates.csv", index=False)
    fits.to_csv(OUTPUT / "wetlab_fits.csv", index=False)
    print(fits[["source", "cell_line", "drug", "ic50_nM", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
