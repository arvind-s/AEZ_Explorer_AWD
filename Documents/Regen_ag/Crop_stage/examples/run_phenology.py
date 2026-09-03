#!/usr/bin/env python3
"""CLI for farm-level paddy phenology assessment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phenology_pipeline.config import PhenologyConfig
from phenology_pipeline.phenology_engine import run_phenology


def main() -> None:
    p = argparse.ArgumentParser(description="Paddy phenology: STAC S1/S2 + ERA5-Land GDD")
    p.add_argument("--polygon", required=True, help="Farm polygon GeoJSON / SHP")
    p.add_argument("--farm-id-col", default="farm_id")
    p.add_argument("--farm-id", default=None)
    p.add_argument("--output-dir", default="phenology_output")

    p.add_argument("--download-start", default="2025-05-01", help="Kharif download start (May 1)")
    p.add_argument("--download-end", default="2025-11-30")
    p.add_argument("--peak-search-end", default="2025-09-30")
    p.add_argument("--assessment-date", default=None)

    p.add_argument("--sowing-date", default=None)
    p.add_argument("--transplant-date", default=None)
    p.add_argument("--date-confidence", choices=["trusted", "uncertain"], default="uncertain")

    p.add_argument("--stac-root", default=None)
    p.add_argument("--stac-cache-dir", default=None, help="Reuse STAC outputs from this directory")
    p.add_argument("--stac-interval", default="12D")
    p.add_argument("--resolution", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-download", action="store_true", help="Use cached STAC only")
    p.add_argument("--force-download", action="store_true", help="Ignore STAC cache")

    p.add_argument(
        "--compute-gdd",
        action="store_true",
        help="Fetch ERA5 weather and compute GDD-based staging (slower)",
    )
    p.add_argument("--era5-cache", default=".cache/phenology/era5")
    p.add_argument(
        "--era5-source",
        choices=["openmeteo", "cds"],
        default="openmeteo",
        help="Weather source for GDD (openmeteo=fast, cds=slow CDS queue)",
    )
    p.add_argument("--variety-days", type=int, default=120)

    args = p.parse_args()

    cfg = PhenologyConfig(
        polygon_path=args.polygon,
        farm_id_col=args.farm_id_col,
        farm_id=args.farm_id,
        output_dir=args.output_dir,
        download_start=args.download_start,
        download_end=args.download_end,
        peak_search_end=args.peak_search_end,
        sowing_date=args.sowing_date,
        transplant_date=args.transplant_date,
        date_confidence=args.date_confidence,
        stac_interval=args.stac_interval,
        stac_resolution_m=args.resolution,
        stac_max_workers=args.workers,
        download_data=not args.no_download,
        timeseries_cache_dir=args.stac_cache_dir,
        era5_cache_dir=args.era5_cache,
        era5_source=args.era5_source,
        compute_gdd=args.compute_gdd,
        variety_duration_days=args.variety_days,
    )
    if args.stac_root:
        cfg.stac_root = Path(args.stac_root)

    if args.date_confidence == "trusted" and not args.transplant_date and not args.sowing_date:
        p.error("--transplant-date or --sowing-date required when --date-confidence=trusted")

    result = run_phenology(
        cfg,
        assessment_date=args.assessment_date,
        force_download=args.force_download,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
