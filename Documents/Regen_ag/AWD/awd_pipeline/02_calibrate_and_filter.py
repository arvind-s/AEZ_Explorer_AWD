#!/usr/bin/env python3
"""
Calibrate + speckle-filter the clipped Sentinel-1 scenes produced by
01_download_s1_grd.py, then aggregate to a per-date AOI backscatter time
series CSV (the input the v0 wetness-index model in 03 consumes).

- sentinel-1-rtc assets: already calibrated to terrain-corrected gamma-naught
  (linear). We just convert to dB and speckle-filter.
- sentinel-1-grd assets: raw amplitude DN. We approximate beta-naught
  calibration using utils.amplitude_dn_to_beta_naught_db. This needs a real
  betaNought LUT constant per scene (from the scene's
  schema-calibration-<pol> XML asset) -- wire that up before trusting the
  GRD path for anything beyond a rough sanity check. See utils.py docstring.

Not executed against real imagery in the build sandbox (no network access to
any satellite data host there) -- see README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from utils import to_db, refined_lee_filter, load_aoi_geojson


def load_and_filter(tif_path: str, aoi_geom: dict, is_rtc: bool) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        arr, _ = rio_mask(src, [aoi_geom], crop=True, nodata=np.nan)
    arr = arr[0].astype(np.float64)

    if is_rtc:
        linear = arr  # already calibrated gamma-naught, linear units
    else:
        # Placeholder: treat raw DN^2 as a proxy linear value. Replace with
        # utils.amplitude_dn_to_beta_naught_db(arr, beta_naught_lut_value)
        # once you've extracted the real per-scene calibration constant.
        linear = arr ** 2

    filtered_linear = refined_lee_filter(np.nan_to_num(linear, nan=np.nanmedian(linear)))
    return to_db(filtered_linear)


def zonal_stats(db_arr: np.ndarray) -> dict:
    valid = db_arr[np.isfinite(db_arr)]
    if valid.size == 0:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
        "n_valid": int(valid.size),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="raw_scenes/manifest.json")
    ap.add_argument("--aoi", default="aoi_sathupalle_wgs84.geojson")
    ap.add_argument("--collection", default="sentinel-1-rtc", choices=["sentinel-1-rtc", "sentinel-1-grd"])
    ap.add_argument("--out", default="s1_aoi_timeseries.csv")
    args = ap.parse_args()

    aoi_geom, _ = load_aoi_geojson(args.aoi)
    is_rtc = args.collection == "sentinel-1-rtc"

    with open(args.manifest) as f:
        manifest = json.load(f)

    rows = []
    for record in manifest:
        for pol, path in record.get("assets", {}).items():
            try:
                db_arr = load_and_filter(path, aoi_geom, is_rtc)
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {path}: {e}")
                continue
            stats = zonal_stats(db_arr)
            rows.append(
                {
                    "date": record["datetime"],
                    "orbit_state": record["orbit_state"],
                    "polarization": pol,
                    **stats,
                }
            )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "orbit_state", "polarization", "mean", "median", "p10", "p90", "n_valid"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
