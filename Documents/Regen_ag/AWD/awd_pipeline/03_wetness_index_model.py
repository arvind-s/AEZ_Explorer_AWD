#!/usr/bin/env python3
"""
v0 AWD model: no labeled training data required.

Takes the per-date AOI VV backscatter time series (from
02_calibrate_and_filter.py) and:
  1. Computes a Lovell (2019)-style Wetness Index (WI), rescaled against the
     series' own min/max (relative, not a fixed global threshold).
  2. Cross-checks with an Otsu dynamic threshold (Tran et al., 2022) on the
     raw dB values as a second opinion on which dates are "flooded".
  3. Detects wet->dry->wet drying cycles and derives an AWD-likelihood score.
  4. Plots the season's WI curve with detected cycles marked.

This is intentionally a heuristic/unsupervised baseline -- see the roadmap
doc (AWD_Global_Model_Roadmap.md) for the planned upgrade path to a pooled,
covariate-conditioned ML sequence model once open regional AWD labels are
confirmed accessible.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime

import numpy as np

from utils import wetness_index, otsu_water_threshold, detect_drying_cycles


def load_series(csv_path: str, polarization: str = "vv"):
    dates, values = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["polarization"] != polarization:
                continue
            if row["mean"] in ("", "nan"):
                continue
            dates.append(datetime.fromisoformat(row["date"]))
            values.append(float(row["mean"]))
    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    values = np.array(values)[order]
    return dates, values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeseries", default="s1_aoi_timeseries.csv")
    ap.add_argument("--polarization", default="vv")
    ap.add_argument("--wet-threshold", type=float, default=0.3)
    ap.add_argument("--dry-threshold", type=float, default=-0.1)
    ap.add_argument("--plot-out", default="awd_wetness_index.png")
    args = ap.parse_args()

    dates, db_values = load_series(args.timeseries, args.polarization)
    if len(dates) < 3:
        raise SystemExit(f"Need at least 3 dates, got {len(dates)}. Check the timeseries CSV.")

    wi = wetness_index(db_values)
    otsu_thresh = otsu_water_threshold(db_values)
    flooded_by_otsu = db_values <= otsu_thresh

    result = detect_drying_cycles(dates, wi, args.wet_threshold, args.dry_threshold)

    print(f"Otsu VV threshold: {otsu_thresh:.2f} dB "
          f"({flooded_by_otsu.sum()}/{len(dates)} dates classified flooded)")
    print(f"Detected drying cycles: {result.n_cycles}")
    for start, end in result.cycle_dates:
        print(f"  cycle: {start.date()} -> {end.date()}")
    print(f"AWD-likelihood score: {result.awd_likelihood:.2f} "
          f"(0 = looks continuously flooded, 1 = looks like well-managed AWD)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, wi, marker="o", label="Wetness Index")
        ax.axhline(args.wet_threshold, color="tab:blue", ls="--", lw=0.8, label="wet threshold")
        ax.axhline(args.dry_threshold, color="tab:orange", ls="--", lw=0.8, label="dry threshold")
        for start, end in result.cycle_dates:
            ax.axvspan(start, end, color="tab:green", alpha=0.15)
        ax.set_ylabel("Wetness Index (-1 dry .. +1 wet)")
        ax.set_title(f"AWD likelihood: {result.awd_likelihood:.2f} | cycles: {result.n_cycles}")
        ax.legend(loc="lower right", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=150)
        print(f"Saved plot: {args.plot_out}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
